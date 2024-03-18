use core::{
    basis::BasisSet,
    config::{ConfigBasisSet, ConfigMolecule},
    molecule::Molecule,
    testing::TestInstance,
};
use std::{error::Error, fs::File};

type Result<T> = std::result::Result<T, Box<dyn Error>>;

struct TestCaseGenerator {
    basis_set_name: &'static str,
    basis_set: BasisSet,
}

impl TestCaseGenerator {
    fn new(basis_set: &'static str) -> Result<Self> {
        Ok(Self {
            basis_set_name: basis_set,
            basis_set: serde_json::from_reader::<_, ConfigBasisSet>(File::open(format!(
                "../data/basis/{basis_set}.json"
            ))?)?
            .try_into()?,
        })
    }

    fn test_case(self, molecule: &str) -> Result<Self> {
        let mol: ConfigMolecule =
            serde_json::from_reader(File::open(format!("../data/mol/{molecule}.json"))?)?;
        let mol: Molecule = mol.into();

        TestInstance::new(
            format!("{molecule} {}", self.basis_set_name),
            &self.basis_set,
            &mol,
        )
        .save(format!(
            "benches/test-inputs/{molecule}_{}.json",
            self.basis_set_name
        ))?;
        Ok(self)
    }
}

#[test]
fn generate_tests() -> Result<()> {
    for basis in ["6-31G", "STO-3G"] {
        let generator = TestCaseGenerator::new(basis)?;

        generator
            .test_case("water")?
            .test_case("hydrogen")?
            .test_case("benzene")?;
    }

    Ok(())
}
