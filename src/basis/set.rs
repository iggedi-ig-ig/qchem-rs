use std::collections::HashMap;

use crate::{atom::Atom, bse::ElementType};

use super::BasisFunctionType;

#[derive(Debug)]
pub(crate) struct BasisSet {
    atomic_mapping: HashMap<ElementType, AtomicBasis>,
}

impl BasisSet {
    /// Returns the basis of a given atom, if it exists.
    pub(crate) fn for_atom(&self, atom: &Atom) -> Option<&AtomicBasis> {
        self.atomic_mapping
            .get(&ElementType::from_ordinal(atom.ordinal)?)
    }

    /// Create a new basis set given mappings from element type to the basis of that element
    pub(crate) fn new(atomic_mapping: HashMap<ElementType, AtomicBasis>) -> Self {
        Self { atomic_mapping }
    }
}

/// Represents the basis functions for a single atom.
#[derive(Debug)]
pub(crate) struct AtomicBasis {
    // TODO: maybe split this up into different shells. This might allow
    //  grouping same shell atoms for faster integration
    pub(crate) basis_functions: Vec<BasisFunctionType>,
}

impl AtomicBasis {
    pub(crate) fn empty() -> Self {
        Self {
            basis_functions: Vec::new(),
        }
    }
}
