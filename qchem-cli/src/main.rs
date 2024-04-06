use core::{
    basis::BasisSet,
    config::{ConfigBasisSet, ConfigMolecule},
    hf::{
        restricted_hartree_fock, unrestricted_hartree_fock, HartreeFockInput,
        MolecularElectronConfig, RestrictedHartreeFockOutput, UnrestrictedHartreeFockOutput,
    },
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
            let basis_set: ConfigBasisSet = serde_json::from_reader(File::open(basis_set)?)?;
            let molecule: ConfigMolecule = serde_json::from_reader(File::open(molecule)?)?;

            let basis_set: BasisSet = basis_set.try_into()?;
            let molecule: Molecule = molecule.into();

            let hf_output = restricted_hartree_fock(&HartreeFockInput {
                molecule: &molecule,
                configuration: MolecularElectronConfig::ClosedShell,
                basis_set: &basis_set,
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
        QcCommand::UnrestrictedHartreeFock {
            basis_set,
            molecule,
            charge,
            spin_multiplicity,
            max_iterations,
            epsilon,
        } => {
            let basis_set: ConfigBasisSet = serde_json::from_reader(File::open(basis_set)?)?;
            let molecule: ConfigMolecule = serde_json::from_reader(File::open(molecule)?)?;

            let basis_set: BasisSet = basis_set.try_into()?;
            let molecule: Molecule = molecule.into();

            let config = match (charge, spin_multiplicity) {
                (0, 0) => MolecularElectronConfig::ClosedShell,
                (0, _) => {
                    return Err(
                        "cannot have non-zero spin multiplicity for an uncharged molecule".into(),
                    )
                }
                (_, 0) => {
                    return Err("charged molecule cannot have spin multiplicity of zero".into())
                }
                (charge, multiplicity) => MolecularElectronConfig::OpenShell {
                    molecular_charge: charge,
                    spin_multiplicity: multiplicity.try_into().unwrap(),
                },
            };

            let hf_output = unrestricted_hartree_fock(&HartreeFockInput {
                molecule: &molecule,
                configuration: config,
                basis_set: &basis_set,
                max_iterations,
                epsilon,
            });

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
                    println!("hartree fock converged after {iterations} iterations");
                    println!("electronic energy: {electronic_energy:3.3}");
                    println!("nuclear repulsion energy: {nuclear_repulsion:3.3}");
                    println!("hartree fock energy: {:3.3}", output.total_energy());
                    println!("orbital energies spin up: {orbital_energies_alpha:3.3?}");
                    println!("orbital energies spin down: {orbital_energies_beta:3.3?}");
                }
                None => panic!("hartree fock did not converge"),
            }
        }
    }

    Ok(())
}
