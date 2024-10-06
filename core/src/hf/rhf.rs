use molint::system::{Atom, MolecularSystem};
use nalgebra::DMatrix;

use crate::diis::Diis;

use super::{utils, HartreeFockConfig};

/// The output of a restricted hartree fock calculation
#[derive(Debug)]
#[non_exhaustive]
#[allow(unused)]
pub struct RestrictedHartreeFockOutput {
    /// the orbital energies that were found in this hartree fock calculation, sorted in
    /// ascending order
    pub orbital_energies: Vec<f64>,
    /// The electronic energy of the system
    pub electronic_energy: f64,
    /// The nuclear repulsion energy
    pub nuclear_repulsion: f64,
    /// After how many iterations did the system converge
    pub iterations: usize,
}

impl RestrictedHartreeFockOutput {
    pub fn total_energy(&self) -> f64 {
        self.electronic_energy + self.nuclear_repulsion
    }
}

pub fn restricted_hartree_fock(
    system: &MolecularSystem,
    config: &HartreeFockConfig,
) -> Option<RestrictedHartreeFockOutput> {
    let n_electrons = system.atoms.iter().map(|a| a.ordinal).sum();
    let n_basis = system.n_basis();

    let nuclear_repulsion = compute_nuclear_repulsion(&system.atoms);

    let overlap = DMatrix::from(molint::overlap(system));
    let kinetic = DMatrix::from(molint::kinetic(system));
    let nuclear = DMatrix::from(molint::nuclear(system));
    let electron = molint::eri(system);

    let core_hamiltonian = kinetic + nuclear;
    let transform = compute_transformation_matrix(&overlap);
    let mut density = compute_hückel_density(
        &core_hamiltonian,
        &overlap,
        &transform,
        n_basis,
        n_electrons,
    );

    let mut electron_terms = vec![0.0; n_basis.pow(4)];
    for (j, i, x, y) in itertools::iproduct!(0..n_basis, 0..n_basis, 0..n_basis, 0..n_basis) {
        electron_terms[j * n_basis.pow(3) + i * n_basis.pow(2) + y * n_basis + x] =
            electron[(i, j, x, y)] - 0.5 * electron[(i, x, j, y)];
    }

    // start of scf iteration
    let mut diis = Diis::new();
    for iteration in 0..=config.max_iterations {
        let electronic_hamiltonian =
            compute_electronic_hamiltonian(&density, &electron_terms, n_basis);

        let fock = &core_hamiltonian + &electronic_hamiltonian;
        let error = &fock * &density * &overlap - &overlap * &density * &fock;

        let fock = diis.fock(error, fock).expect("DIIS failed");
        let transformed_fock = &transform.transpose() * (&fock * &transform);
        let (transformed_coefficients, obrital_energies) = utils::sorted_eigs(transformed_fock);
        let coefficients = &transform * &transformed_coefficients;

        let new_density = compute_updated_density(&coefficients, n_basis, n_electrons);

        const F: f64 = 1.0;
        let density_change = new_density - &density;
        density += &density_change * F;

        let electronic_energy =
            0.5 * (&density * (2.0 * &core_hamiltonian + &electronic_hamiltonian)).trace();

        let density_rms =
            (density_change.map_diagonal(|entry| entry.powi(2)).sum() / n_basis as f64).sqrt();

        log::info!(
            "iteration {iteration:<4} - electronic energy {electronic_energy:1.4}. density rms {density_rms:1.4e}",
        );

        if density_rms < config.epsilon {
            let electronic_energy =
                0.5 * (&density * (2.0 * &core_hamiltonian + &electronic_hamiltonian)).trace();
            return Some(RestrictedHartreeFockOutput {
                orbital_energies: obrital_energies.as_slice().to_vec(),
                electronic_energy,
                nuclear_repulsion,
                iterations: iteration,
            });
        }
    }

    // TODO: finish implementation
    None
}

fn compute_nuclear_repulsion(atoms: &[Atom]) -> f64 {
    let n_atoms = atoms.len();

    let mut potential = 0.0;
    for atom_a in 0..n_atoms {
        for atom_b in atom_a + 1..n_atoms {
            potential += (atoms[atom_a].ordinal * atoms[atom_b].ordinal) as f64
                / (atoms[atom_b].position - atoms[atom_a].position).norm()
        }
    }
    log::debug!("nulcear repulsion energy: {potential}");
    potential
}

fn compute_transformation_matrix(overlap: &DMatrix<f64>) -> DMatrix<f64> {
    let (u, _) = utils::eigs(overlap.clone());
    let diagonal_matrix = &u.transpose() * (overlap * &u);

    let diagonal_inv_sqrt =
        DMatrix::from_diagonal(&diagonal_matrix.map_diagonal(|f| f.sqrt().recip()));
    &u * (diagonal_inv_sqrt * &u.transpose())
}

fn compute_hückel_density(
    hamiltonian: &DMatrix<f64>,
    overlap: &DMatrix<f64>,
    transform: &DMatrix<f64>,
    n_basis: usize,
    n_electrons: usize,
) -> DMatrix<f64> {
    const WOLFSBERG_HELMHOLTZ: f64 = 1.75;
    let hamiltonian_eht = utils::symmetric_matrix(n_basis, |i, j| {
        WOLFSBERG_HELMHOLTZ * overlap[(i, j)] * (hamiltonian[(i, i)] + hamiltonian[(j, j)]) / 2.0
    });

    let transformed = &transform.transpose() * (hamiltonian_eht * transform);
    let (coeffs_prime, _orbital_energies) = utils::sorted_eigs(transformed);
    let coeffs = transform * coeffs_prime;

    compute_updated_density(&coeffs, n_basis, n_electrons)
}

fn compute_electronic_hamiltonian(
    density: &DMatrix<f64>,
    electron_terms: &[f64],
    n_basis: usize,
) -> DMatrix<f64> {
    utils::symmetric_matrix(n_basis, |i, j| {
        let mut sum = 0.0;
        for y in 0..n_basis {
            for x in 0..n_basis {
                sum += density[(x, y)]
                    * electron_terms[j * n_basis.pow(3) + i * n_basis.pow(2) + y * n_basis + x];
            }
        }
        sum
    })
}

fn compute_updated_density(
    coefficients: &DMatrix<f64>,
    n_basis: usize,
    n_electrons: usize,
) -> DMatrix<f64> {
    utils::symmetric_matrix(n_basis, |i, j| {
        let mut sum = 0.0;
        for k in 0..n_electrons / 2 {
            sum += coefficients[(i, k)] * coefficients[(j, k)]
        }
        2.0 * sum
    })
}
