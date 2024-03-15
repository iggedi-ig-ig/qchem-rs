use nalgebra::Vector3;

/// Represents an atom in a molecule.
#[derive(Copy, Clone, Debug, PartialEq)]
pub(crate) struct Atom {
    pub(crate) position: Vector3<f64>,
    pub(crate) ordinal: usize,
    // TODO: ionic charge? Or do we only store that as a molecule property?
    // TODO: at some point, mass might become necessary aswell
}

impl Atom {
    pub(crate) fn charge(&self) -> i32 {
        self.ordinal as i32
    }
}
