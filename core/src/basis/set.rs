use std::collections::HashMap;

use crate::{atom::Atom, periodic_table::ElementType};

use super::ContractedGaussian;

#[derive(Debug)]
pub struct BasisSet {
    atomic_mapping: HashMap<ElementType, AtomicBasis>,
}

impl BasisSet {
    /// Returns the basis of a given atom, if it exists.
    pub fn for_atom(&self, atom: &Atom) -> Option<&AtomicBasis> {
        self.atomic_mapping.get(&atom.element_type)
    }

    /// Create a new basis set given mappings from element type to the basis of that element
    pub(crate) fn new(atomic_mapping: HashMap<ElementType, AtomicBasis>) -> Self {
        Self { atomic_mapping }
    }
}

/// Represents the basis functions for a single atom.
#[derive(Debug)]
pub struct AtomicBasis {
    pub(crate) shells: Vec<ElectronShell>,
}

impl AtomicBasis {
    pub(crate) fn empty() -> Self {
        Self { shells: Vec::new() }
    }

    pub fn basis_functions(&self) -> impl Iterator<Item = &ContractedGaussian> {
        self.shells.iter().flat_map(|shell| &shell.basis_functions)
    }
}

#[derive(Debug, Clone)]
#[allow(unused)]
pub(crate) struct ElectronShell {
    pub(crate) angular_magnitude: i32,
    pub(crate) basis_functions: Vec<ContractedGaussian>,
}

impl ElectronShell {
    pub(crate) fn new(angular_magnitude: i32) -> Self {
        Self {
            angular_magnitude,
            basis_functions: Vec::new(),
        }
    }
}
