use nalgebra::Vector3;
use smallvec::SmallVec;

#[derive(Copy, Clone, Debug, PartialEq)]
/// Function of the form K*x^i*y^j*z^k*exp(-alpha*x^2)
pub struct Gaussian {
    pub(crate) exponent: f64,
    /// The coefficient of this gaussian and optionally the normalization constant
    pub(crate) coefficient: f64,
    /// (i, j, k) exponents of polynomial terms
    pub(crate) angular: (i32, i32, i32),
}

#[derive(Clone, Debug, PartialEq)]
pub struct ContractedGaussian(pub(crate) SmallVec<[Gaussian; 6]>);

#[derive(Clone, Debug, PartialEq)]
pub enum BasisFunctionType {
    ContractedGaussian(ContractedGaussian),
}

pub struct BasisFunction {
    /// The type of basis function this basis function has
    pub(crate) function_type: BasisFunctionType,
    /// The position of this basis function, in natural units
    pub(crate) position: Vector3<f64>,
}
