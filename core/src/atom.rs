use nalgebra::Vector3;

use crate::periodic_table::ElementType;

/// Represents an atom in a molecule.
#[derive(Copy, Clone, Debug, PartialEq)]
pub(crate) struct Atom {
    pub(crate) position: Vector3<f64>,
    pub(crate) element_type: ElementType,
    // TODO: ionic charge? Or do we only store that as a molecule property?
    // TODO: at some point, mass might become necessary aswell
}

impl Atom {
    pub(crate) fn charge(&self) -> i32 {
        self.element_type as i32
    }
}
