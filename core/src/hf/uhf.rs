use molint::{
    storage::EriTensor,
    system::{Atom, MolecularSystem},
};
use nalgebra::{DMatrix, DVector};

use crate::diis::Diis;

use super::{utils, HartreeFockConfig};

/// The output of a restricted hartree fock calculation
#[derive(Debug)]
#[non_exhaustive]
#[allow(unused)]
pub struct UnrestrictedHartreeFockOutput {
    /// the spin up orbital energies that were found in this hartree fock calculation, sorted in
    /// ascending order
    pub orbital_energies_alpha: Vec<f64>,
    /// the spin down orbital energies that were found in this hartree fock calculation, sorted in
    /// ascending order
    pub orbital_energies_beta: Vec<f64>,
    /// The electronic energy of the system
    pub electronic_energy: f64,
    /// The nuclear repulsion energy
    pub nuclear_repulsion: f64,
    /// After how many iterations did the system converge
    pub iterations: usize,
}

impl UnrestrictedHartreeFockOutput {
    pub fn total_energy(&self) -> f64 {
        self.electronic_energy + self.nuclear_repulsion
    }
}

pub fn unrestricted_hartree_fock(
    system: &MolecularSystem,
    config: &HartreeFockConfig,
) -> Option<UnrestrictedHartreeFockOutput> {
    let n_basis = system.n_basis();

    // TODO: completeness
    let n_electrons: usize = system.atoms.iter().map(|a| a.ordinal).sum();
    let n_alpha = n_electrons / 2;
    let n_beta = n_electrons / 2;

    let nuclear_repulsion = compute_nuclear_repulsion(&system.atoms);
    log::debug!("nulcear repulsion energy: {nuclear_repulsion}");

    // TODO: do we need to pre-calculate all of the integrals? I don't think, for example, all ERI integrals are actually used.
    //  if we could skip some of them, that would be a huge performance gain.
    let overlap = DMatrix::from(molint::overlap(system));
    let kinetic = DMatrix::from(molint::kinetic(system));
    let nuclear = DMatrix::from(molint::nuclear(system));
    let electron = molint::eri(system);

    let core_hamiltonian = kinetic + nuclear;
    let transform = compute_transformation_matrix(&overlap);

    let mut density_alpha =
        compute_hückel_density_guess(&core_hamiltonian, &overlap, &transform, n_basis, n_alpha);
    let mut density_beta =
        compute_hückel_density_guess(&core_hamiltonian, &overlap, &transform, n_basis, n_beta);

    let mut electronic_hamiltonians = [
        DMatrix::zeros(n_basis, n_basis),
        DMatrix::zeros(n_basis, n_basis),
    ];
    let mut coefficient_matrices = [
        DMatrix::zeros(n_basis, n_basis),
        DMatrix::zeros(n_basis, n_basis),
    ];
    let mut orbital_energies = [DVector::zeros(n_basis), DVector::zeros(n_basis)];

    // start of scf iteration
    let min_size = 2;
    let max_size = 8;
    let mut diis = [Diis::new(min_size, max_size), Diis::new(min_size, max_size)];

    for iteration in 0..=config.max_iterations {
        for spin in 0..=1 {
            // "main" density and "other" density
            let (density_one, density_two) = match spin {
                0 => (&mut density_alpha, &density_beta),
                1 => (&mut density_beta, &density_alpha),
                _ => unreachable!(),
            };
            let diis = &mut diis[spin];

            let electronic_hamiltonian =
                compute_electronic_hamiltonian(density_one, density_two, &electron, n_basis);

            let fock = &core_hamiltonian + &electronic_hamiltonian;
            let error = &fock * &*density_one * &overlap - &overlap * &*density_one * &fock;
            let fock = diis
                .fock(error, fock)
                .unwrap_or_else(|| panic!("DIIS failed in spin {spin}"));

            electronic_hamiltonians[spin] = electronic_hamiltonian;

            let transformed_fock = &transform.transpose() * (&fock * &transform);
            let (transformed_coefficients, spin_orbital_energies) =
                utils::sorted_eigs(transformed_fock);
            let coefficients = &transform * &transformed_coefficients;

            coefficient_matrices[spin] = coefficients;
            orbital_energies[spin] = spin_orbital_energies;
        }

        // second loop, because we need the new coefficients to compute the new density matrices
        let mut density_rms = 0.0;
        for spin in 0..=1 {
            // "main" density and "other" density
            let (old_density, coefficients, electrons) = match spin {
                0 => (&mut density_alpha, &coefficient_matrices[0], n_alpha),
                1 => (&mut density_beta, &coefficient_matrices[1], n_beta),
                _ => unreachable!(),
            };

            let new_density = compute_updated_density(coefficients, n_basis, electrons);

            const F: f64 = 1.0;
            let density_change = &new_density - &*old_density;
            *old_density += &density_change * F;

            let self_rms =
                (density_change.map_diagonal(|entry| entry.powi(2)).sum() / n_basis as f64).sqrt();

            density_rms += self_rms;

            log::debug!(
                "spin {} - density rms {self_rms:03.3e}",
                ["up", "down"][spin]
            )
        }

        let density_rms = density_rms / 2.0;
        log::info!("iteration {iteration} - density rms {density_rms:03.3e}");
        if density_rms / 2.0 < config.epsilon {
            let [orbital_energies_alpha, orbital_energies_beta] =
                orbital_energies.map(|x| x.as_slice().to_vec());
            let [electronic_hamiltonian_alpha, electronic_hamiltonian_beta] =
                electronic_hamiltonians;

            let energy_alpha = 0.5
                * (&density_alpha * (2.0 * &core_hamiltonian + &electronic_hamiltonian_alpha))
                    .trace();

            let energy_beta = 0.5
                * (&density_beta * (2.0 * &core_hamiltonian + &electronic_hamiltonian_beta))
                    .trace();

            let electronic_energy = energy_alpha + energy_beta;

            return Some(UnrestrictedHartreeFockOutput {
                orbital_energies_alpha,
                orbital_energies_beta,
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
    potential
}

fn compute_transformation_matrix(overlap: &DMatrix<f64>) -> DMatrix<f64> {
    let (u, _) = utils::eigs(overlap.clone());
    let diagonal_matrix = &u.transpose() * (overlap * &u);

    let diagonal_inv_sqrt =
        DMatrix::from_diagonal(&diagonal_matrix.map_diagonal(|f| f.sqrt().recip()));
    &u * (diagonal_inv_sqrt * &u.transpose())
}

fn compute_hückel_density_guess(
    hamiltonian: &DMatrix<f64>,
    overlap: &DMatrix<f64>,
    transform: &DMatrix<f64>,
    n_basis: usize,
    n_occupied: usize,
) -> DMatrix<f64> {
    const WOLFSBERG_HELMHOLTZ: f64 = 1.75;
    let hamiltonian_eht = utils::symmetric_matrix(n_basis, |i, j| {
        WOLFSBERG_HELMHOLTZ * overlap[(i, j)] * (hamiltonian[(i, i)] + hamiltonian[(j, j)]) / 2.0
    });

    let transformed = &transform.transpose() * (hamiltonian_eht * transform);
    let (coeffs_prime, _orbital_energies) = utils::sorted_eigs(transformed);
    let coeffs = transform * coeffs_prime;

    compute_updated_density(&coeffs, n_basis, n_occupied)
}

fn compute_electronic_hamiltonian(
    density_one: &DMatrix<f64>,
    density_two: &DMatrix<f64>,
    multi: &EriTensor,
    n_basis: usize,
) -> DMatrix<f64> {
    utils::symmetric_matrix(n_basis, |i, j| {
        let mut sum = 0.0;
        for k in 0..n_basis {
            for l in 0..n_basis {
                sum += density_one[(k, l)] * multi[(i, j, k, l)]
                    + density_two[(k, l)] * multi[(i, j, k, l)]
                    - density_one[(k, l)] * multi[(i, k, j, l)]
            }
        }
        sum
    })
}

fn compute_updated_density(
    coefficients: &DMatrix<f64>,
    n_basis: usize,
    n_occupied: usize,
) -> DMatrix<f64> {
    utils::symmetric_matrix(n_basis, |i, j| {
        let mut sum = 0.0;
        for k in 0..n_occupied {
            sum += coefficients[(i, k)] * coefficients[(j, k)];
        }
        sum
    })
}
