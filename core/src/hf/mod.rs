mod mo;
pub mod rhf;
mod uhf;
pub(super) mod utils;

use std::num::NonZeroU32;

pub use rhf::{restricted_hartree_fock, RestrictedHartreeFockOutput};
pub use uhf::{unrestricted_hartree_fock, UnrestrictedHartreeFockOutput};

use crate::{
    basis::{BasisFunction, BasisSet},
    molecule::Molecule,
};

/// The state of a molecule.
pub enum MolecularElectronConfig {
    ClosedShell,
    OpenShell {
        molecular_charge: i32,
        spin_multiplicity: NonZeroU32,
    },
}

/// The input to a hartree fock calculation
pub struct HartreeFockInput<'a> {
    /// the molecule to run hartree fock for
    pub molecule: &'a Molecule,
    /// the concfiguration of the molecule
    pub configuration: MolecularElectronConfig,
    /// what basis set to use
    pub basis_set: &'a BasisSet,
    /// the maximum number of iterations to try
    pub max_iterations: usize,
    /// the smallest number that isn't treated as zero. For example, if the density
    /// matrix rms changes by less than this, the system is considered converged.
    pub epsilon: f64,
}

impl HartreeFockInput<'_> {
    pub(crate) fn basis(&self) -> Vec<BasisFunction> {
        // TODO: group integral terms by similar terms?
        self.molecule
            .atoms
            .iter()
            .flat_map(|atom| {
                let atomic_basis = self
                    .basis_set
                    .for_atom(atom)
                    .unwrap_or_else(|| panic!("no basis for element {:?}", atom.element_type));

                atomic_basis
                    .basis_functions()
                    .map(|function_type| BasisFunction {
                        contracted_gaussian: function_type.clone(),
                        position: atom.position,
                    })
            })
            .collect::<Vec<_>>()
    }

    /// Returns the number of total electrons in the system
    pub(crate) fn n_electrons(&self) -> usize {
        let base_electron_count = self
            .molecule
            .atoms
            .iter()
            .map(|atom| atom.element_type as usize)
            .sum::<usize>();

        match self.configuration {
            MolecularElectronConfig::ClosedShell => base_electron_count,
            MolecularElectronConfig::OpenShell {
                molecular_charge, ..
            } => base_electron_count.saturating_add_signed(-molecular_charge as isize),
        }
    }

    /// Returns the number of electrons in the alpha (by convention, spin up) state
    pub(crate) fn n_alpha(&self) -> usize {
        match self.configuration {
            MolecularElectronConfig::ClosedShell => self.n_electrons() / 2,
            MolecularElectronConfig::OpenShell {
                spin_multiplicity, ..
            } => {
                (self
                    .n_electrons()
                    .saturating_sub(spin_multiplicity.get() as usize)
                    + 1)
                    / 2
            }
        }
    }

    /// Returns the number of electrons in the beta (by convention, spin down) state
    pub(crate) fn n_beta(&self) -> usize {
        match self.configuration {
            MolecularElectronConfig::ClosedShell => self.n_electrons() / 2,
            MolecularElectronConfig::OpenShell {
                spin_multiplicity, ..
            } => {
                (self
                    .n_electrons()
                    .saturating_add(spin_multiplicity.get() as usize)
                    - 1)
                    / 2
            }
        }
    }
}
