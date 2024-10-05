use core::hf::{
    restricted_hartree_fock, HartreeFockInput, MolecularElectronConfig, RestrictedHartreeFockOutput,
};
use std::{error::Error, path::PathBuf};

use clap::{ArgAction, Parser, Subcommand};
use molint::{
    basis::BasisSet,
    system::{Atom, MolecularSystem},
};
use nalgebra::Point3;

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
    #[command(name = "rhf")]
    RestrictedHartreeFock {
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
        QcCommand::RestrictedHartreeFock {
            basis_set,
            molecule,
            max_iterations,
            epsilon,
        } => {
            let basis_set: BasisSet = BasisSet::load(basis_set)?;
            let molecule = MolecularSystem::from_atoms(
                &[
                    Atom {
                        ordinal: 1,
                        position: Point3::new(0.4175, 0.0, 0.83),
                    },
                    Atom {
                        ordinal: 8,
                        position: Point3::new(0.0, 0.0, -0.31),
                    },
                    Atom {
                        ordinal: 1,
                        position: Point3::new(-0.4175, 0.0, 0.83),
                    },
                ],
                &basis_set,
            );

            let hf_output = restricted_hartree_fock(&HartreeFockInput {
                system: molecule,
                configuration: MolecularElectronConfig::ClosedShell,
                max_iterations,
                epsilon,
            });

            match hf_output {
                Some(
                    ref output @ RestrictedHartreeFockOutput {
                        ref orbital_energies,
                        electronic_energy,
                        nuclear_repulsion,
                        iterations,
                        ..
                    },
                ) => {
                    println!("hartree fock converged after {iterations} iterations");
                    println!("electronic energy: {electronic_energy:3.3}");
                    println!("nuclear repulsion energy: {nuclear_repulsion:3.3}");
                    println!("hartree fock energy: {:3.3}", output.total_energy());
                    println!("orbital energies: {orbital_energies:3.3?}");
                }
                None => panic!("hartree fock did not converge"),
            }
        }
    }

    Ok(())
}
