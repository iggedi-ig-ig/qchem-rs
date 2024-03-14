use std::collections::VecDeque;

use nalgebra::{DMatrix, DVector, SymmetricEigen};

use crate::{
    atom::Atom,
    basis::BasisFunction,
    hf::mo::MolecularOrbitals,
    integrals::{DefaultIntegrator, ElectronTensor, Integrator},
};

use super::{HartreeFockInput, HartreeFockOutput};

pub fn hartree_fock(input: &HartreeFockInput) -> Option<HartreeFockOutput> {
    // exchangable integrator
    let integrator = DefaultIntegrator::default();

    // TODO: group integral terms by similar terms?
    let basis = input
        .molecule
        .atoms
        .iter()
        .flat_map(|atom| {
            let atomic_basis = input
                .basis_set
                .for_atom(atom)
                .unwrap_or_else(|| panic!("no basis for atom with ordinal {}", atom.ordinal));

            atomic_basis
                .basis_functions
                .iter()
                .map(|function_type| BasisFunction {
                    function_type: function_type.clone(),
                    position: atom.position,
                })
        })
        .collect::<Vec<_>>();

    let n_basis = basis.len();

    let n_electrons = input
        .molecule
        .atoms
        .iter()
        .map(|atom| atom.ordinal as i32 - atom.ion_charge)
        .sum::<i32>() as usize;

    let _nuclear_repulsion = calculate_nuclear_repulsion(&input.molecule.atoms);

    // TODO: do we need to pre-calculate all of the integrals? I don't think, for example, all ERI integrals are actually used.
    //  if we could skip some of them, that would be a huge performance gain.
    let overlap = calculate_overlap_matrix(&basis, &integrator);
    let kinetic = calculate_kinetic_matrix(&basis, &integrator);
    let nuclear = calculate_nuclear_matrix(&basis, &input.molecule.atoms, &integrator);
    let electron = ElectronTensor::from_basis(&basis, &integrator);

    let core_hamiltonian = kinetic + nuclear;
    let transform = calculate_transformation_matrix(&overlap);
    let mut density = calculate_hückel_density(
        &core_hamiltonian,
        &overlap,
        &transform,
        n_basis,
        n_electrons,
    );

    let mut electron_terms = vec![0.0; n_basis.pow(4)];
    for (j, i, x, y) in itertools::iproduct!(0..n_basis, 0..n_basis, 0..n_basis, 0..n_basis) {
        electron_terms[j * n_basis.pow(3) + i * n_basis.pow(2) + y * n_basis + x] =
            electron[(i, j, x, y)] - 0.5 * electron[(i, x, j, y)];
    }

    // start of scf iteration
    for iteration in 0..=input.max_iterations {
        let density_guess = calculate_density_guess(&density, &electron_terms, n_basis);

        let fock = &core_hamiltonian + &density_guess;

        let transformed_fock = &transform.transpose() * (&fock * &transform);
        let (transformed_coefficients, obrital_energies) = sorted_eigs(transformed_fock);
        let coefficients = &transform * &transformed_coefficients;

        let new_density = calculate_updated_density(&coefficients, n_basis, n_electrons);

        // TODO: f-mixing
        let density_change = new_density - &density;
        density += &density_change;

        let electronic_energy =
            0.5 * (&density * (2.0 * &core_hamiltonian + &density_guess)).trace();

        let density_rms =
            (density_change.map_diagonal(|entry| entry.powi(2)).sum() / n_basis as f64).sqrt();

        log::info!(
            "iteration {iteration} - electronic energy {electronic_energy}. density rms {density_rms}",
        );

        if density_rms < input.epsilon {
            return Some(HartreeFockOutput {
                orbitals: MolecularOrbitals::from_coefficient_matrix(&coefficients),
                basis,
                orbital_energies: obrital_energies.as_slice().to_vec(),
            });
        }
    }

    // TODO: finish implementation
    None
}

fn calculate_nuclear_repulsion(atoms: &[Atom]) -> f64 {
    log::debug!("calculating nuclear repulsion");

    let n_atoms = atoms.len();

    let mut potential = 0.0;
    for atom_a in 0..n_atoms {
        for atom_b in atom_a + 1..n_atoms {
            potential += (atoms[atom_a].ion_charge * atoms[atom_b].ion_charge) as f64
                / (atoms[atom_b].position - atoms[atom_a].position).norm()
        }
    }
    potential
}

// TODO: the basis argument type is a bit awkward to work with
fn calculate_overlap_matrix(
    basis: &[BasisFunction],
    integrator: &dyn Integrator<Function = BasisFunction>,
) -> DMatrix<f64> {
    log::debug!("calculating overlap integrals");

    hermitian(basis.len(), |i, j| {
        dbg!(integrator.overlap((&basis[i], &basis[j])))
    })
}

// TODO: the basis argument type is a bit awkward to work with
fn calculate_kinetic_matrix(
    basis: &[BasisFunction],
    integrator: &dyn Integrator<Function = BasisFunction>,
) -> DMatrix<f64> {
    log::debug!("calculating kinetic energy integrals");

    hermitian(basis.len(), |i, j| {
        dbg!(integrator.kinetic((&basis[i], &basis[j])))
    })
}

// TODO: the basis argument type is a bit awkward to work with
fn calculate_nuclear_matrix(
    basis: &[BasisFunction],
    nuclei: &[Atom],
    integrator: &dyn Integrator<Function = BasisFunction>,
) -> DMatrix<f64> {
    log::debug!("calculating electron-nuclear attraction energy integrals");

    hermitian(basis.len(), |i, j| {
        dbg!(integrator.nuclear((&basis[i], &basis[j]), nuclei))
    })
}

fn calculate_transformation_matrix(overlap: &DMatrix<f64>) -> DMatrix<f64> {
    let (u, _) = eigs(overlap.clone());
    let diagonal_matrix = &u.transpose() * (overlap * &u);

    let diagonal_inv_sqrt =
        DMatrix::from_diagonal(&diagonal_matrix.map_diagonal(|f| f.sqrt().recip()));
    &u * (diagonal_inv_sqrt * &u.transpose())
}

fn calculate_hückel_density(
    hamiltonian: &DMatrix<f64>,
    overlap: &DMatrix<f64>,
    transform: &DMatrix<f64>,
    n_basis: usize,
    n_electrons: usize,
) -> DMatrix<f64> {
    let hamiltonian_eht = hermitian(n_basis, |i, j| {
        0.875 * overlap[(i, j)] * (hamiltonian[(i, i)] + hamiltonian[(j, j)])
    });

    let transformed = &transform.transpose() * (hamiltonian_eht * transform);
    let (coeffs_prime, _orbital_energies) = sorted_eigs(transformed);
    let coeffs = transform * coeffs_prime;

    hermitian(n_basis, |i, j| {
        2.0 * (0..n_electrons / 2).fold(0.0, |acc, k| acc + coeffs[(i, k)] * coeffs[(j, k)])
    })
}

fn calculate_density_guess(
    density: &DMatrix<f64>,
    electron_terms: &[f64],
    n_basis: usize,
) -> DMatrix<f64> {
    hermitian(n_basis, |i, j| {
        (0..n_basis).fold(0.0, |acc, y| {
            acc + (0..n_basis).fold(0.0, |acc, x| {
                acc + density[(x, y)]
                    * electron_terms[j * n_basis.pow(3) + i * n_basis.pow(2) + y * n_basis + x]
            })
        })
    })
}

fn calculate_updated_density(
    coefficients: &DMatrix<f64>,
    n_basis: usize,
    n_electrons: usize,
) -> DMatrix<f64> {
    hermitian(n_basis, |i, j| {
        2.0 * (0..n_electrons / 2).fold(0.0, |acc, k| {
            acc + coefficients[(i, k)] * coefficients[(j, k)]
        })
    })
}

fn hermitian(n: usize, mut func: impl FnMut(usize, usize) -> f64) -> DMatrix<f64> {
    let m = DMatrix::from_fn(n, n, |i, j| if i <= j { func(i, j) } else { 0.0 });
    DMatrix::from_fn(n, n, |i, j| if i <= j { m[(i, j)] } else { m[(j, i)] })
}

fn eigs(matrix: DMatrix<f64>) -> (DMatrix<f64>, DVector<f64>) {
    let eigs = SymmetricEigen::new(matrix);
    (eigs.eigenvectors, eigs.eigenvalues)
}

fn sorted_eigs(matrix: DMatrix<f64>) -> (DMatrix<f64>, DVector<f64>) {
    let (eigenvectors, eigenvalues) = eigs(matrix);

    let mut val_vec_pairs = eigenvalues
        .into_iter()
        .zip(eigenvectors.column_iter())
        .collect::<Vec<_>>();

    val_vec_pairs.sort_unstable_by(|(a, _), (b, _)| a.total_cmp(b));

    let (values, vectors): (Vec<_>, Vec<_>) = val_vec_pairs.into_iter().unzip();

    (
        DMatrix::from_columns(&vectors),
        DVector::from_column_slice(&values),
    )
}

// TODO: reimplement diis
fn _diis(
    error_vectors: &VecDeque<DMatrix<f64>>,
    fock_matricies: &VecDeque<DMatrix<f64>>,
) -> Option<DMatrix<f64>> {
    assert_eq!(error_vectors.len(), fock_matricies.len());
    let n = error_vectors.len();

    let mut matrix = DMatrix::zeros(n + 1, n + 1);
    // upper block
    for (i, j) in itertools::iproduct!(0..n, 0..n) {
        matrix[(i, j)] = error_vectors[i].dot(&error_vectors[j]);
    }

    // last row
    for i in 0..n {
        matrix[(n, i)] = -1.0;
    }

    // last col
    for i in 0..n {
        matrix[(i, n)] = -1.0;
    }

    // last entry
    matrix[(n, n)] = 0.0;

    let mut b = DVector::zeros(n + 1);
    b[(n, 0)] = -1.0;

    matrix.try_inverse().map(|inv| inv * b).map(|c| {
        c.iter()
            .enumerate()
            .take(n)
            .map(|(i, &x)| x * &fock_matricies[i])
            .sum()
    })
}
