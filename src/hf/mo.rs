use nalgebra::{DMatrix, Vector3};

use crate::basis::BasisFunction;

/// Represents the molecular orbitals of a hartree fock calculation
#[derive(Debug)]
pub(crate) struct MolecularOrbitals {
    orbitals: Vec<MolecularOrbital>,
}

impl MolecularOrbitals {
    /// anything less than this is considered zero
    const ZERO_CUTOFF: f64 = 1e-4;

    /// Reconstruct the molecular orbitals from a converged coefficient matrix and a basis
    pub(crate) fn from_coefficient_matrix(coefficient_matrix: &DMatrix<f64>) -> Self {
        let mut orbitals = Vec::with_capacity(coefficient_matrix.ncols());

        for column in coefficient_matrix.column_iter() {
            let (indices, elements) = column
                .iter()
                .enumerate()
                .filter(|(_, element)| element.abs() > Self::ZERO_CUTOFF)
                .unzip::<_, _, Vec<_>, Vec<_>>();

            orbitals.push(MolecularOrbital {
                basis_functions: indices,
                coefficients: elements,
            });
        }

        Self { orbitals }
    }

    /// Evaluate the n-th lowest energy orbital at a given positon
    pub(crate) fn evaluate_orbital(
        &self,
        basis: &[BasisFunction],
        orbital: usize,
        position: Vector3<f64>,
    ) -> f64 {
        let MolecularOrbital {
            ref basis_functions,
            ref coefficients,
        } = self.orbitals[orbital];

        let mut output = 0.0;

        for (&basis_function_index, &coeff) in basis_functions.iter().zip(coefficients) {
            output += coeff * basis[basis_function_index].evaluate(position);
        }

        output
    }
}

type BasisFunctionId = usize;

#[derive(Debug)]
struct MolecularOrbital {
    basis_functions: Vec<BasisFunctionId>,
    coefficients: Vec<f64>,
}
