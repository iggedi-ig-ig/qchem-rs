use std::hash::Hash;
use std::ops::Index;

use crate::basis::BasisFunction;

use super::Integrator;

/// An integral index used in the two-electron integrals of a basis set.
///
/// The index represents the four indices (x, y, z, w) used to calculate a two-electron integral:
///   int_{x,y,z,w} = int_{xy|zw} = <x y | z w>
///
/// Since two-electron integrals are symmetric in (xy) and (zw), this struct stores its indices
/// with the correct order: xy <= zw, reducing the total number of unique integrals to compute.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub(crate) struct IntegralIndex(usize, usize, usize, usize);

impl IntegralIndex {
    /// Creates a new integral index with the given indices.
    pub(crate) const fn new(index: (usize, usize, usize, usize)) -> Self {
        let (i, j, k, l) = Self::correct_order(index);
        Self(i, j, k, l)
    }

    /// Returns the indices with the correct order, such that xy <= zw.
    #[inline(always)]
    const fn correct_order(
        (i, j, k, l): (usize, usize, usize, usize),
    ) -> (usize, usize, usize, usize) {
        let (i, j) = if i < j { (i, j) } else { (j, i) };
        let (k, l) = if k < l { (k, l) } else { (l, k) };

        let ij = i * (i + 1) / 2 + j;
        let kl = k * (k + 1) / 2 + l;

        if ij < kl {
            (i, j, k, l)
        } else {
            (k, l, i, j)
        }
    }

    pub fn linear(&self, size: usize) -> usize {
        let &Self(i, j, k, l) = self;
        l * size.pow(3) + k * size.pow(2) + j * size + i
    }
}

impl std::fmt::Display for IntegralIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let &Self(i, j, k, l) = self;
        write!(f, "({} {}|{} {})", i, j, k, l)
    }
}

/// An electron tensor representing electron-electron repulsion integrals between
/// four contracted Gaussian functions in a given basis set.
pub struct ElectronTensor {
    data: Vec<f64>,
    /// side length
    size: usize,
}

impl ElectronTensor {
    /// Constructs an `ElectronTensor` from the given basis set. Computes electron-electron
    /// repulsion integrals for each unique combination of four Gaussian functions in the basis
    /// set and stores them in a hashmap. This method utilizes parallel processing with the
    /// Rayon library to speed up computation time.
    ///
    /// # Arguments
    ///
    /// * `basis` - A slice of `ContractedGaussian` functions representing the basis set to use
    /// for computing electron-electron repulsion integrals.
    ///
    /// # Returns
    ///
    /// An `ElectronTensor` containing a hashmap of electron-electron repulsion integrals
    /// computed for each unique combination of four Gaussian functions in the given basis set.
    pub fn from_basis(
        basis: &[BasisFunction],
        integrator: &impl Integrator<Item = BasisFunction>,
    ) -> Self {
        // Initialize variables for computing the total number of integrals and a thread-safe
        // container for storing the resulting electron-electron repulsion integrals.
        let n_basis = basis.len();
        let mut data = vec![0.0; n_basis.pow(4)];

        // compute diagonal first - we need these entries for screening
        for i in 0..n_basis {
            for j in i..n_basis {
                let index = IntegralIndex(i, j, i, j);
                let linear = index.linear(n_basis);

                data[linear] =
                    integrator.electron_repulsion((&basis[i], &basis[j], &basis[i], &basis[j]));
            }
        }

        // symmetries:
        // (ij|kl) = (ji|lk) = (kl|ij) = (lk|ij) = (kj|il) = (li|jk) = (il|kj) = (jk|li)

        let cap = (n_basis - 1) * (n_basis) * (n_basis + 1).pow(2) / 6;
        let mut to_compute = Vec::with_capacity(cap);

        for i in 0..n_basis {
            for j in i..n_basis {
                for k in 0..n_basis {
                    for l in k..n_basis {
                        // Enforce symmetry constraints + skip diagonal
                        if i < k || j < l {
                            to_compute.push(IntegralIndex(i, j, k, l));
                        }
                    }
                }
            }
        }

        to_compute.sort_unstable_by_key(|index| index.linear(n_basis));

        #[cfg(feature = "rayon")]
        {
            use rayon::iter::{ParallelBridge, ParallelIterator};

            to_compute
                .chunks(512)
                .par_bridge()
                .map(|indices| {
                    let mut output = Vec::with_capacity(indices.len());
                    for index @ &IntegralIndex(x, y, z, w) in indices {
                        let linear = index.linear(n_basis);
                        let integral = integrator
                            .electron_repulsion((&basis[x], &basis[y], &basis[z], &basis[w]));

                        log::trace!("ERI {index} = {integral:<1.3}");
                        output.push((linear, integral))
                    }
                    output
                })
                .collect::<Vec<_>>() // iterators are lazy - we collect to evaluate all elements
                .into_iter()
                .flatten()
                .for_each(|(index, integral)| data[index] = integral);
        }

        #[cfg(not(feature = "rayon"))]
        to_compute
            .into_iter()
            .for_each(|index @ IntegralIndex(x, y, z, w)| {
                let linear = index.linear(n_basis);
                let integral =
                    integrator.electron_repulsion((&basis[x], &basis[y], &basis[z], &basis[w]));

                log::trace!("ERI {index} = {integral:<1.8}");
                data[linear] = integral;
            });

        Self {
            data,
            size: n_basis,
        }
    }
}

// TODO: readd this
fn _cauchy_schwarz_estimate(data: &[Option<f64>], n_basis: usize, integral: IntegralIndex) -> f64 {
    let IntegralIndex(x, y, z, w) = integral;

    let diagonal_index_ij = IntegralIndex::new((x, y, x, y));
    let diagonal_index_kl = IntegralIndex::new((z, w, z, w));

    f64::sqrt(
        data[diagonal_index_ij.linear(n_basis)].expect("diagonal is set")
            * data[diagonal_index_kl.linear(n_basis)].expect("diagonal is set"),
    )
}

impl Index<(usize, usize, usize, usize)> for ElectronTensor {
    type Output = f64;

    fn index(&self, index: (usize, usize, usize, usize)) -> &Self::Output {
        let index = IntegralIndex::new(index);
        let linear = index.linear(self.size);
        &self.data[linear]
    }
}

impl Index<IntegralIndex> for ElectronTensor {
    type Output = f64;

    fn index(&self, index: IntegralIndex) -> &Self::Output {
        let linear = index.linear(self.size);
        &self.data[linear]
    }
}
