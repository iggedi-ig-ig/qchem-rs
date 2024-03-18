pub mod atom;
pub mod basis;
pub mod config;
pub mod hf;
pub mod integrals;
pub mod molecule;
pub mod periodic_table;

pub mod testing {
    use std::{error::Error, fs::File, path::Path};

    use serde::{Deserialize, Serialize};

    use crate::{
        basis::{BasisFunction, BasisSet},
        molecule::Molecule,
    };

    #[derive(Serialize, Deserialize)]
    pub struct TestInstance {
        basis_functions: Vec<BasisFunction>,
    }

    impl TestInstance {
        pub fn new(basis_set: &BasisSet, molecule: &Molecule) -> Self {
            let mut basis_functions = Vec::new();

            for atom in molecule.atoms() {
                let atomic_basis = basis_set.for_atom(atom).unwrap();

                basis_functions.extend(atomic_basis.basis_functions().map(|function_type| {
                    BasisFunction {
                        function_type: function_type.clone(),
                        position: atom.position,
                    }
                }))
            }

            Self { basis_functions }
        }

        pub fn save(&self, path: impl AsRef<Path>) -> Result<(), Box<dyn Error>> {
            Ok(serde_json::to_writer(
                File::options().create(true).write(true).open(path)?,
                self,
            )?)
        }

        pub fn basis_functions(&self) -> &[BasisFunction] {
            &self.basis_functions
        }
    }
}
