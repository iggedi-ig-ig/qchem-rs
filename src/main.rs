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

            match hf_output {
                Some(
                    ref output @ HartreeFockOutput {
                        ref orbital_energies,
                        electronic_energy,
                        nuclear_repulsion,
                        ..
                    },
                ) => {
                    log::info!("hartree fock converged");
                    log::info!("electronic energy: {electronic_energy:3.3}");
                    log::info!("nuclear repulsion energy: {nuclear_repulsion:3.3}");
                    log::info!("hartree fock energy: {:3.3}", output.total_energy());
                    log::info!("orbital energies: {orbital_energies:3.3?}");
                }
                None => log::error!("hartree fock did not converge"),
            }
        }
    }

    Ok(())
}
