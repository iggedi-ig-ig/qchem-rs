use nalgebra::Vector3;

use crate::periodic_table::ElementType;

/// Represents an atom in a molecule.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Atom {
    pub(crate) position: Vector3<f64>,
    pub(crate) element_type: ElementType,
}

impl Atom {
    /// Returns the charge of this nucleus 
    pub fn nuclear_charge(&self) -> i32 {
        self.element_type as i32
    }

    pub fn position(&self) -> &Vector3<f64> {
        &self.position
    }
}
