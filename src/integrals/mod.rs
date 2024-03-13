use crate::Atom;

pub mod naive;

pub type DefaultIntegrator = naive::McMurchieDavidson;

pub trait Integrator {
    type Function;

    /// Calculate the overlap integral between two basis functions.
    fn overlap(&mut self, functions: (&Self::Function, &Self::Function)) -> f64;

    /// Calculate the kinetic energy integral between two basis functions.
    fn kinetic(&mut self, functions: (&Self::Function, &Self::Function)) -> f64;

    /// Calculate the nuclear attraction integral between two basis functions and the nuclei of a quantum system.
    fn nuclear(&mut self, functions: (&Self::Function, &Self::Function), nuclei: &[Atom]) -> f64;

    /// Calculate the electron-electron repulsion integral between four basis functions.
    fn electron_repulsion(
        &mut self,
        functions: (
            &Self::Function,
            &Self::Function,
            &Self::Function,
            &Self::Function,
        ),
    ) -> f64;
}
