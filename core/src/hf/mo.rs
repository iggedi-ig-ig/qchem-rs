// TODO: implement slater determinants to actually work with the wave function.
#![allow(unused)]
use std::env::consts::FAMILY;

use log::Record;
use nalgebra::{DMatrix, Vector2, Vector3};

use molint::basis::ContractedGaussian;
/// Represents the molecular orbitals of a hartree fock calculation
#[derive(Debug)]
pub(crate) struct MolecularOrbitals {
    orbitals: Vec<MolecularOrbital>,
}

impl MolecularOrbitals {
    /// anything less than this is considered zero
    const ZERO_CUTOFF: f64 = 1e-4;

    /// Reconstruct the molecular orbitals from a converged coefficient matrix and a basis
    pub(crate) fn from_matrix(coefficient_matrix: &DMatrix<f64>) -> Self {
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
}

type BasisFunctionId = usize;

#[derive(Debug)]
struct MolecularOrbital {
    basis_functions: Vec<BasisFunctionId>,
    coefficients: Vec<f64>,
}

fn slater_normalization(n: usize) -> f64 {
    (2..=n).map(|i| i as f64).product::<f64>().sqrt()
}
