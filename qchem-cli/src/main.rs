use core::hf::{
    restricted_hartree_fock, unrestricted_hartree_fock, HartreeFockConfig,
    RestrictedHartreeFockOutput, UnrestrictedHartreeFockOutput,
};
use std::{error::Error, path::PathBuf, time::Instant};

use clap::{ArgAction, Parser, Subcommand};
use molint::{basis::BasisSet, system::MolecularSystem};

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
    #[command(name = "uhf")]
    UnrestrictedHartreeFock {
        /// What basis set to use for the hartree fock calculation
        #[arg(long, short)]
        basis_set: PathBuf,
        /// A path to the molecule to perform the calculation on
        #[arg(long, short)]
        molecule: PathBuf,
        /// The charge of the molecule
        #[arg(long, short, default_value_t = 0)]
        charge: i32,
        /// The spin multiplicity of the molecule
        #[arg(long, short, default_value_t = 0)]
        spin_multiplicity: u32,
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
            let basis_set = BasisSet::load(basis_set)?;
            let system = MolecularSystem::load(molecule, &basis_set)?;

            let start = Instant::now();
            let hf_output = restricted_hartree_fock(
                &system,
                &HartreeFockConfig {
                    max_iterations,
                    epsilon,
                },
            );

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
                    println!(
                        "hartree fock converged after {iterations} iterations and {:0.2?}",
                        start.elapsed()
                    );
                    println!("electronic energy: {electronic_energy:3.3}");
                    println!("nuclear repulsion energy: {nuclear_repulsion:3.3}");
                    println!("hartree fock energy: {:3.3}", output.total_energy());
                    println!("orbital energies: {orbital_energies:3.3?}");
                }
                None => panic!("hartree fock did not converge"),
            }
        }

        // TODO (completeness): charge and spin multiplicity
        QcCommand::UnrestrictedHartreeFock {
            basis_set,
            molecule,
            charge: _,
            spin_multiplicity: _,
            max_iterations,
            epsilon,
        } => {
            let basis_set = BasisSet::load(basis_set)?;
            let system = MolecularSystem::load(molecule, &basis_set)?;

            let start = Instant::now();
            let hf_output = unrestricted_hartree_fock(
                &system,
                &HartreeFockConfig {
                    max_iterations,
                    epsilon,
                },
            );

            match hf_output {
                Some(
                    ref output @ UnrestrictedHartreeFockOutput {
                        ref orbital_energies_alpha,
                        ref orbital_energies_beta,
                        electronic_energy,
                        nuclear_repulsion,
                        iterations,
                        ..
                    },
                ) => {
                    println!(
                        "hartree fock converged after {iterations} iterations and {:?}",
                        start.elapsed()
                    );
                    println!("electronic energy: {electronic_energy:3.3}");
                    println!("nuclear repulsion energy: {nuclear_repulsion:3.3}");
                    println!("hartree fock energy: {:3.3}", output.total_energy());
                    println!("orbital energies alpha spin:   {orbital_energies_alpha:3.3?}");
                    println!("orbital energies beta spin: {orbital_energies_beta:3.3?}");
                }
                None => panic!("hartree fock did not converge"),
            }
        }
    }

    Ok(())
}
