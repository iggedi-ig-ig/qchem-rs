use crate::atom::Atom;

/// Represents a molecule
#[derive(Debug)]
pub(crate) struct Molecule {
    pub(crate) atoms: Vec<Atom>,
}
