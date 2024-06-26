pub(crate) use electron_tensor::ElectronTensor;

use crate::atom::Atom;

pub mod electron_tensor;
pub mod mmd;
mod utils;

/// The default integrator
pub(crate) type DefaultIntegrator = mmd::McMurchieDavidson;

pub trait Integrator: Send + Sync {
    type Item;

    /// Calculate the overlap integral between two basis functions.
    fn overlap(&self, functions: (&Self::Item, &Self::Item)) -> f64;

    /// Calculate the kinetic energy integral between two basis functions.
    fn kinetic(&self, functions: (&Self::Item, &Self::Item)) -> f64;

    /// Calculate the nuclear attraction integral between two basis functions and the nuclei of a quantum system.
    fn nuclear(&self, functions: (&Self::Item, &Self::Item), nuclei: &[Atom]) -> f64;

    /// Calculate the electron-electron repulsion integral between four basis functions.
    fn electron_repulsion(
        &self,
        functions: (&Self::Item, &Self::Item, &Self::Item, &Self::Item),
    ) -> f64;
}
