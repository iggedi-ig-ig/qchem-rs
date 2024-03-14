use crate::Atom;
pub(crate) use electron_tensor::ElectronTensor;

mod electron_tensor;
mod mmd;

/// The default integrator
pub(crate) type DefaultIntegrator = mmd::McMurchieDavidson;

pub(crate) trait Integrator {
    type Function;

    /// Calculate the overlap integral between two basis functions.
    fn overlap(&self, functions: (&Self::Function, &Self::Function)) -> f64;

    /// Calculate the kinetic energy integral between two basis functions.
    fn kinetic(&self, functions: (&Self::Function, &Self::Function)) -> f64;

    /// Calculate the nuclear attraction integral between two basis functions and the nuclei of a quantum system.
    fn nuclear(&self, functions: (&Self::Function, &Self::Function), nuclei: &[Atom]) -> f64;

    /// Calculate the electron-electron repulsion integral between four basis functions.
    fn electron_repulsion(
        &self,
        functions: (
            &Self::Function,
            &Self::Function,
            &Self::Function,
            &Self::Function,
        ),
    ) -> f64;
}
