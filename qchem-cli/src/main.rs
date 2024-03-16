use core::{
    basis::BasisSet,
    config::{ConfigBasisSet, ConfigMolecule},
    hf::{hartree_fock, HartreeFockInput, HartreeFockOutput},
    molecule::Molecule,
};
use std::{error::Error, fs::File, path::PathBuf};

use clap::{ArgAction, Parser, Subcommand};

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
        /// What basis set to use for the hartree fock calculation
        #[arg(long, short)]
        basis_set: PathBuf,
        /// A path to the molecule to perform the calculation on
        #[arg(long, short)]
        molecule: PathBuf,
        /// The maximum number of iterations the SCF loop should attempt before the
        /// system is considered to not diverge
        #[arg(long, default_value_t = 100)]
        max_iterations: usize,
        /// if the rms of the density matrix drops below this, the system is considered
        /// converged
        #[arg(long, default_value_t = 1e-6)]
        epsilon: f64,
    },
}

fn main() -> Result<(), Box<dyn Error>> {
    pretty_env_logger::init();

    let args: Args = Args::parse();

    match args.command {
        QcCommand::HartreeFock {
            basis_set,
            molecule,
            max_iterations,
            epsilon,
        } => {
            let basis_set: ConfigBasisSet = serde_json::from_reader(File::open(basis_set)?)?;
            let molecule: ConfigMolecule = serde_json::from_reader(File::open(molecule)?)?;

            let basis_set: BasisSet = basis_set.try_into()?;
            let molecule: Molecule = molecule.into();

            let hf_output = hartree_fock(&HartreeFockInput {
                molecule: &molecule,
                basis_set: &basis_set,
                max_iterations,
                epsilon,
            });

            match hf_output {
                Some(
                    ref output @ HartreeFockOutput {
                        ref orbital_energies,
                        electronic_energy,
                        nuclear_repulsion,
                        iterations,
                        ..
                    },
                ) => {
                    log::info!("hartree fock converged after {iterations} iterations");
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