mod mo;
pub mod rhf;
pub(super) mod utils;

use std::num::NonZeroU32;

use molint::system::MolecularSystem;
pub use rhf::{restricted_hartree_fock, RestrictedHartreeFockOutput};

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
    pub system: MolecularSystem<'a>,
    /// the concfiguration of the molecule
    pub configuration: MolecularElectronConfig,
    /// the maximum number of iterations to try
    pub max_iterations: usize,
    /// the smallest number that isn't treated as zero. For example, if the density
    /// matrix rms changes by less than this, the system is considered converged.
    pub epsilon: f64,
}

impl HartreeFockInput<'_> {
    /// Returns the number of total electrons in the system
    pub(crate) fn n_electrons(&self) -> usize {
        let base_electron_count = self
            .system
            .atoms
            .iter()
            .map(|atom| atom.ordinal)
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
