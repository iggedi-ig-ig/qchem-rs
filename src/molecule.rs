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
            let &[x, y, z] = atom.position.as_slice() else {
                // TODO: error handling
                panic!("Atom didn't have x, y, z coordinates")
            };

            atoms.push(Atom {
                position: Vector3::new(x, y, z),
                ordinal: atom.element as usize,
            });
        }

        Self { atoms }
    }
}
