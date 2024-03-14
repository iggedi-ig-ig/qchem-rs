use std::{error::Error, fs::File, path::PathBuf};

use atom::Atom;
use basis::BasisSet;
use bse::BseBasisSet;
use clap::{ArgAction, Parser, Subcommand};
use hf::{hartree_fock, HartreeFockInput, HartreeFockOutput};
use molecule::{ConfigMolecule, Molecule};

mod atom;
mod basis;
mod bse;
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
    #[command(name = "hf")]
    HartreeFock {
        basis_set: PathBuf,
        molecule: PathBuf,
    },
}

fn main() -> Result<(), Box<dyn Error>> {
    pretty_env_logger::init();

    let args: Args = Args::parse();

    match args.command {
        QcCommand::HartreeFock {
            basis_set,
            molecule,
        } => {
            let basis_set: BseBasisSet = serde_json::from_reader(File::open(basis_set)?)?;
            let molecule: ConfigMolecule = serde_json::from_reader(File::open(molecule)?)?;

            let basis_set: BasisSet = basis_set.try_into()?;
            let molecule: Molecule = molecule.into();

            let hf_input = HartreeFockInput {
                molecule: &molecule,
                basis_set: &basis_set,
                max_iterations: 100,
                epsilon: 1e-12,
            };

            let hf_output = hartree_fock(&hf_input);

            if let Some(HartreeFockOutput {
                orbital_energies, ..
            }) = hf_output
            {
                println!("Hartree fock converged");
                println!("orbital energies {orbital_energies:?}");
            } else {
                println!("Hartree fock didn't didn't converge")
            }
        }
    }

    Ok(())
}
