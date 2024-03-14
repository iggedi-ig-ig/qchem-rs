use std::collections::HashMap;

use atom::Atom;
use basis::{BasisFunctionType, ContractedGaussian, Gaussian};
use clap::{ArgAction, Parser, Subcommand};
use nalgebra::Vector3;
use smallvec::smallvec;

use crate::{
    basis::{AtomicBasis, BasisSet},
    hf::HartreeFockInput,
    molecule::Molecule,
};

mod atom;
mod basis;
mod hf;
mod integrals;
mod molecule;
mod utils;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[command(subcommand)]
    command: QcCommand,

    #[arg(long, short, action=ArgAction::SetFalse)]
    verbose: bool,
}

#[derive(Subcommand, Debug)]
enum QcCommand {
    HartreeFock {/* specify what to output */},
}

fn main() {
    pretty_env_logger::init();

    // let args: Args = Args::parse();

    let test_molecule = Molecule::new(vec![
        Atom {
            position: Vector3::zeros(),
            ordinal: 1,
            ion_charge: 0,
            mass: 1.0,
        },
        Atom {
            position: Vector3::new(0.0, 0.0, 1.0),
            ordinal: 1,
            ion_charge: 0,
            mass: 1.0,
        },
    ]);

    let test_basis_set = BasisSet {
        atomic_mapping: HashMap::from([(
            1,
            AtomicBasis {
                basis_functions: vec![BasisFunctionType::ContractedGaussian(ContractedGaussian(
                    smallvec![Gaussian {
                        exponent: 1.0,
                        coefficient: Gaussian::norm(1.0, (0, 0, 0)),
                        angular: (0, 0, 0)
                    }],
                ))],
            },
        )]),
    };

    let test_input = HartreeFockInput {
        molecule: &test_molecule,
        basis_set: &test_basis_set,
        max_iterations: 100,
        epsilon: 1e-6,
    };

    let test_output = hf::hartree_fock(&test_input);

    println!("{:?}", test_output.unwrap().orbital_energies);
}
