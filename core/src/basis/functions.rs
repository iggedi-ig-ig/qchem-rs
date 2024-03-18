use nalgebra::Vector3;
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;

/// Function of the form K*x^i*y^j*z^k*exp(-alpha*x^2)
#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Gaussian {
    pub exponent: f64,
    /// The coefficient of this gaussian and optionally the normalization constant
    pub coefficient: f64,
    /// (i, j, k) exponents of polynomial terms
    pub angular: (i32, i32, i32),
}

impl Gaussian {
    pub fn norm(exponent: f64, angular: (i32, i32, i32)) -> f64 {
        let (i, j, k) = angular;

        (std::f64::consts::FRAC_2_PI * exponent)
            .powi(3)
            .sqrt()
            .sqrt()
            * f64::sqrt(
                (8.0 * exponent).powi(i + j + k)
                    / ((i + 1..=2 * i).product::<i32>()
                        * (j + 1..=2 * j).product::<i32>()
                        * (k + 1..=2 * k).product::<i32>()) as f64,
            )
    }
}

/// Linear combination of many [`Gaussian`]s
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ContractedGaussian(pub SmallVec<[Gaussian; 6]>);

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BasisFunction {
    /// The type of basis function this basis function has
    pub contracted_gaussian: ContractedGaussian,
    /// The position of this basis function, in natural units
    pub position: Vector3<f64>,
}

impl BasisFunction {
    /// Evaluate this basis function at a given position
    pub(crate) fn evaluate(&self, _at: Vector3<f64>) -> f64 {
        todo!("implement basis function evaluation")
    }
}
