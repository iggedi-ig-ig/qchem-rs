use crate::atom::Atom;

/// Represents a molecule
#[derive(Debug)]
pub struct Molecule {
    pub(crate) atoms: Vec<Atom>,
}

impl Molecule {
    pub fn atoms(&self) -> &[Atom] {
        &self.atoms
    }
}
