use core::hf::{
    restricted_hartree_fock, HartreeFockInput, MolecularElectronConfig, RestrictedHartreeFockOutput,
};
use std::{error::Error, path::PathBuf};

use approx::{assert_abs_diff_eq, relative_eq};
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

    let basis_set: BasisSet = BasisSet::load("data/basis/6-31G.json")?;
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
        max_iterations: 100,
        epsilon: 1e-6,
    });

    match hf_output {
        Some(RestrictedHartreeFockOutput {
            ref orbital_energies,
            electronic_energy,
            ..
        }) => {
            assert_abs_diff_eq!(electronic_energy, -89.56219148143138, epsilon = 1e-5);

            const ORBITAL_ENERGIES: [f64; 13] = [
                -20.530046784284785,
                -1.858642963156469,
                -0.7310062353124118,
                -0.711834352868239,
                -0.5419965534828293,
                0.20299276202600472,
                0.36032431407863025,
                1.028505309073029,
                1.148278285162504,
                1.1825266842623154,
                1.5183562374176907,
                1.668128908540347,
                1.9522623427206827,
            ];
            assert!(
                orbital_energies
                    .iter()
                    .enumerate()
                    .all(|(i, &f)| relative_eq!(f, ORBITAL_ENERGIES[i], epsilon = 1e-5)),
                "orbital energies didn't stay the same'"
            );
        }
        None => panic!("hartree fock did not converge"),
    }

    Ok(())
}
