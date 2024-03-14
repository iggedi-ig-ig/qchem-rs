use std::{collections::HashMap, error::Error};

use itertools::iproduct;
use serde::Deserialize;
use smallvec::SmallVec;

use crate::basis::{AtomicBasis, BasisFunctionType, BasisSet, ContractedGaussian, Gaussian};

use super::atomic_table::ElementType;

#[derive(Deserialize)]
pub(crate) struct BseBasisSet {
    elements: HashMap<ElementType, ElectronicConfiguration>,
}

impl TryFrom<BseBasisSet> for BasisSet {
    // TODO: use a "better" type for error
    type Error = Box<dyn Error>;

    fn try_from(value: BseBasisSet) -> Result<Self, Self::Error> {
        let mut atomic_mapping = HashMap::with_capacity(value.elements.len());

        // TODO: this is pretty deeply nested, this can definitely be improved somehow
        for (element, configuration) in value.elements {
            let mut element_atomic_basis = AtomicBasis::empty();

            for electron_shell in &configuration.electron_shells {
                let mut basis_functions = Vec::with_capacity(8);

                for (index, &angular_magnitude) in
                    electron_shell.angular_momentum.iter().enumerate()
                {
                    let angular_vectors = generate_angular_vectors(angular_magnitude);

                    for angular in angular_vectors {
                        let mut primitives =
                            SmallVec::with_capacity(electron_shell.exponents.len());

                        for (exponent, coefficient) in electron_shell
                            .exponents
                            .iter()
                            .zip(&electron_shell.coefficients[index])
                        {
                            let exponent = exponent.parse::<f64>()?;
                            let coefficient = coefficient.parse::<f64>()?;

                            let norm = Gaussian::norm(exponent, angular);

                            println!("{angular:?} {exponent} {coefficient} {norm}");

                            primitives.push(Gaussian {
                                exponent,
                                coefficient: coefficient * norm,
                                angular,
                            });
                        }

                        element_atomic_basis.basis_functions.push(
                            BasisFunctionType::ContractedGaussian(ContractedGaussian(primitives)),
                        );
                    }
                }

                // TODO: group by angular term / shell
                element_atomic_basis
                    .basis_functions
                    .append(&mut basis_functions);
            }

            atomic_mapping.insert(element, element_atomic_basis);
        }

        Ok(Self::new(atomic_mapping))
    }
}

// generate all (i, j, k) such that i + j + k = angular
fn generate_angular_vectors(angular_magnitude: i32) -> Vec<(i32, i32, i32)> {
    let mut angular_vectors = Vec::with_capacity(8);

    for (i, j, k) in iproduct!(
        0..=angular_magnitude,
        0..=angular_magnitude,
        0..=angular_magnitude
    ) {
        if i + j + k == angular_magnitude {
            angular_vectors.push((i, j, k));
        }
    }

    angular_vectors
}

#[derive(Deserialize)]
struct ElectronicConfiguration {
    electron_shells: Vec<ElectronShell>,
}

#[derive(Deserialize)]
struct ElectronShell {
    function_type: String,
    angular_momentum: Vec<i32>,
    exponents: Vec<String>,
    coefficients: Vec<Vec<String>>,
}
