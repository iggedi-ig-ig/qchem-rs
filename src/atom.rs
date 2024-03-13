use nalgebra::Vector3;

/// Represents an atom in a molecule.
pub(crate) struct Atom {
    pub(crate) position: Vector3<f64>,
    pub(crate) ordinal: usize,
    pub(crate) ion_charge: i32,
    pub(crate) mass: f64,
}
