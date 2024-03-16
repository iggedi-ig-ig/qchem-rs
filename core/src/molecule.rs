use crate::atom::Atom;

/// Represents a molecule
#[derive(Debug)]
pub struct Molecule {
    pub(crate) atoms: Vec<Atom>,
}
