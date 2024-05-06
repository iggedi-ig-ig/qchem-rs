use nalgebra::{DMatrix, DVector};
use std::collections::VecDeque;

use crate::hf::utils;

const MIN_LENGTH: usize = 5;
const MAX_LENGTH: usize = 12;

struct Sample {
    error: DMatrix<f64>,
    fock: DMatrix<f64>,
}

pub(crate) struct Diis {
    /// (Error, Vector)
    previous_samples: VecDeque<Sample>,
}

impl Diis {
    pub fn new() -> Self {
        Self {
            previous_samples: VecDeque::new(),
        }
    }

    /// _should_ not return None
    pub fn fock(&mut self, error: DMatrix<f64>, fock: DMatrix<f64>) -> Option<DMatrix<f64>> {
        self.previous_samples.push_front(Sample { error, fock });
        self.previous_samples.truncate(MAX_LENGTH);

        let n = self.previous_samples.len();
        if n < MIN_LENGTH {
            return self
                .previous_samples
                .front()
                .map(|Sample { fock, .. }| fock.to_owned());
        }
        let matrix = utils::symmetric_matrix(n + 1, |i, j| match (i, j) {
            (i, j) if i == n && j == n => 0.0,
            (i, j) if i == n || j == n => 1.0,
            (i, j) => self.previous_samples[i]
                .error
                .dot(&self.previous_samples[j].error),
        });

        let b = DVector::from_fn(n + 1, |i, _| if i == n { 1.0 } else { 0.0 });

        let qr = matrix.qr();
        let solution = qr.solve(&b)?;
        Some(
            solution
                .view((0, 0), (n, 1))
                .map_with_location(|i, _, x| x * &self.previous_samples[i].fock)
                .iter()
                .sum(),
        )
    }
}
