use nalgebra::Vector3;
use serde::Deserialize;

use crate::{atom::Atom, bse::ElementType};

/// Represents a molecule
#[derive(Debug)]
pub(crate) struct Molecule {
    pub(crate) atoms: Vec<Atom>,
}

#[derive(Deserialize)]
pub(crate) struct ConfigMolecule(Vec<ConfigAtom>);

#[derive(Deserialize)]
pub(crate) struct ConfigAtom {
    element: ElementType,
    position: Vec<f64>,
}

impl From<ConfigMolecule> for Molecule {
    fn from(value: ConfigMolecule) -> Self {
        let ConfigMolecule(config_atoms) = value;

        let mut atoms = Vec::with_capacity(config_atoms.len());

        for atom in config_atoms {
            atoms.push(Atom {
                position: Vector3::from_column_slice(&atom.position),
                ordinal: atom.element as usize,
                ion_charge: 0,
            });
        }

        Self { atoms }
    }
}
