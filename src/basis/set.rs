use std::collections::HashMap;

use crate::atom::Atom;

use super::BasisFunctionType;

type AtomicOrdinal = usize;

pub(crate) struct BasisSet {
    pub(crate) atomic_mapping: HashMap<AtomicOrdinal, AtomicBasis>,
}

impl BasisSet {
    pub(crate) fn for_atom(&self, atom: &Atom) -> Option<&AtomicBasis> {
        self.atomic_mapping.get(&atom.ordinal)
    }
}

/// Represents the basis functions for a single atom.
pub(crate) struct AtomicBasis {
    // TODO: maybe split this up into different shells. This might allow
    //  grouping same shell atoms for faster integration
    pub(crate) basis_functions: Vec<BasisFunctionType>,
}
