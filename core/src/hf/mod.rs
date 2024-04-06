mod mo;
pub mod scf;
pub(super) mod utils;

use std::num::NonZeroU32;

pub use scf::restricted_hartree_fock;

use crate::{
    basis::{BasisFunction, BasisSet},
    molecule::Molecule,
};

use self::mo::MolecularOrbitals;

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

/// The output of a hartree fock calculation
#[derive(Debug)]
#[non_exhaustive]
pub struct HartreeFockOutput {
    /// The molecular orbitals that were found in the hartree fock calculation.
    /// These are sorted by ascending order in energy.
    pub(crate) orbitals: MolecularOrbitals,
    /// the basis that was used in the hartree fock calculation. This is necessary
    /// to be able to for example evaluate the molecular orbitals that were found
    pub(crate) basis: Vec<BasisFunction>,
    /// the orbital energies that were found in this hartree fock calculation, sorted in
    /// ascending order
    pub orbital_energies: Vec<f64>,
    /// The electronic energy of the system
    pub electronic_energy: f64,
    /// The nuclear repulsion energy
    pub nuclear_repulsion: f64,
    /// After how many iterations did the system converge
    pub iterations: usize,
}

impl HartreeFockOutput {
    pub fn total_energy(&self) -> f64 {
        self.electronic_energy + self.nuclear_repulsion
    }
}
