use std::collections::VecDeque;

use nalgebra::{DMatrix, DVector, SymmetricEigen};

#[inline(always)]
/// Create a symmetric, square matrix. Function is only run for upper triangle of the matrix
pub(crate) fn symmetric_matrix(
    n: usize,
    mut func: impl FnMut(usize, usize) -> f64,
) -> DMatrix<f64> {
    let m = DMatrix::from_fn(n, n, |i, j| if i <= j { func(i, j) } else { 0.0 });
    DMatrix::from_fn(n, n, |i, j| if i <= j { m[(i, j)] } else { m[(j, i)] })
}

pub(super) fn eigs(matrix: DMatrix<f64>) -> (DMatrix<f64>, DVector<f64>) {
    let eigs = SymmetricEigen::new(matrix);
    (eigs.eigenvectors, eigs.eigenvalues)
}

pub(super) fn sorted_eigs(matrix: DMatrix<f64>) -> (DMatrix<f64>, DVector<f64>) {
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
