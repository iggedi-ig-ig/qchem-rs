use std::{collections::HashMap, error::Error};

use serde::Deserialize;
use smallvec::SmallVec;

use crate::{
    basis::{
        AtomicBasis, BasisFunctionType, BasisSet, ContractedGaussian, ElectronShell, Gaussian,
    },
    periodic_table::ElementType,
};

#[derive(Deserialize)]
pub(crate) struct ConfigBasisSet {
    elements: HashMap<ElementType, ConfigElectronicConfiguration>,
}

#[derive(Deserialize)]
struct ConfigElectronicConfiguration {
    electron_shells: Vec<ConfigElectronShell>,
}

#[derive(Deserialize)]
#[allow(unused)]
struct ConfigElectronShell {
    function_type: String,
    angular_momentum: Vec<i32>,
    exponents: Vec<String>,
    coefficients: Vec<Vec<String>>,
}

impl TryFrom<ConfigBasisSet> for BasisSet {
    // TODO: use a "better" type for error
    type Error = Box<dyn Error>;

    fn try_from(value: ConfigBasisSet) -> Result<Self, Self::Error> {
        let mut atomic_mapping = HashMap::with_capacity(value.elements.len());

        // TODO: this is pretty deeply nested, this can definitely be improved somehow
        for (element, configuration) in value.elements {
            let mut element_atomic_basis = AtomicBasis::empty();

            for electron_shell in &configuration.electron_shells {
                for (index, &angular_magnitude) in
                    electron_shell.angular_momentum.iter().enumerate()
                {
                    let mut shell = ElectronShell::new(angular_magnitude);
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

                            primitives.push(Gaussian {
                                exponent,
                                coefficient: coefficient * norm,
                                angular,
                            });
                        }

                        shell
                            .basis_functions
                            .push(BasisFunctionType::ContractedGaussian(ContractedGaussian(
                                primitives,
                            )));
                    }

                    element_atomic_basis.shells.push(shell);
                }
            }

            atomic_mapping.insert(element, element_atomic_basis);
        }

        Ok(Self::new(atomic_mapping))
    }
}

// generate all (i, j, k) such that i + j + k = angular
fn generate_angular_vectors(angular_magnitude: i32) -> Vec<(i32, i32, i32)> {
    let mut angular_vectors = Vec::with_capacity(8);

    for (i, j, k) in itertools::iproduct!(
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
