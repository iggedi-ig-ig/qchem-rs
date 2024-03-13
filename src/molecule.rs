use crate::atom::Atom;

/// Represents a molecule 
pub(crate) struct Molecule {
    atoms: Vec<Atom>,
    molecule_charge: i32,
}

impl Molecule {
    pub fn new(atoms: Vec<Atom>) -> Self {
        Self {
            molecule_charge: atoms.iter().map(|atom| atom.ion_charge).sum(),
            atoms,
        }
    }
}
