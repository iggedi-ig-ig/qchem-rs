mod mo;
pub mod scf;
pub(super) mod utils;

pub use scf::hartree_fock;

use crate::{
    basis::{BasisFunction, BasisSet},
    molecule::Molecule,
};

use self::mo::MolecularOrbitals;

/// The input to a hartree fock calculation
pub struct HartreeFockInput<'a> {
    /// the molecule to run hartree fock for
    pub molecule: &'a Molecule,
    /// what basis set to use
    pub basis_set: &'a BasisSet,
    /// the maximum number of iterations to try
    pub max_iterations: usize,
    /// the smallest number that isn't treated as zero. For example, if the density
    /// matrix rms changes by less than this, the system is considered converged.
    pub epsilon: f64,
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
