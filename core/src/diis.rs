use nalgebra::{DMatrix, DVector};
use std::collections::VecDeque;

struct Sample<T> {
    error: T,
    vector: T,
}

pub(crate) struct Diis<T> {
    /// (Error, Vector)
    previous_samples: VecDeque<Sample<T>>,
}

impl Diis<DMatrix<f64>> {
    pub fn new() -> Self {
        Self {
            previous_samples: VecDeque::new(),
        }
    }

    /// _should_ not return None
    pub fn fock(&mut self, error: DMatrix<f64>, vector: DMatrix<f64>) -> Option<DMatrix<f64>> {
        self.previous_samples.push_front(Sample { error, vector });
        self.previous_samples.truncate(12);

        let n = self.previous_samples.len();
        if n < 5 {
            return self
                .previous_samples
                .front()
                .map(|Sample { vector, .. }| vector.to_owned());
        }

        let matrix = DMatrix::from_fn(n + 1, n + 1, |i, j| match (i, j) {
            (i, j) if i == n && j == n => 0.0,
            (i, j) if i == n || j == n => -1.0,
            _ => self.previous_samples[i]
                .error
                .dot(&self.previous_samples[j].error),
        });

        let mut b = DVector::zeros(n + 1);
        b[(n, 0)] = -1.0;

        let qr = matrix.qr();
        let solution = qr.solve(&b)?;
        Some(
            solution
                .iter()
                .enumerate()
                .take(n)
                .map(|(i, &x)| x * &self.previous_samples[i].vector)
                .sum(),
        )
    }
}
