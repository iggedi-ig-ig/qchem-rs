use crate::atom::Atom;

/// Represents a molecule
pub(crate) struct Molecule {
    pub(crate) atoms: Vec<Atom>,
}

impl Molecule {
    pub(crate) fn new(atoms: Vec<Atom>) -> Self {
        Self { atoms }
    }
}
