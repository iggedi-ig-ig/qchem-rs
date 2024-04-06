use nalgebra::{DMatrix, DVector};

use crate::{
    atom::Atom,
    basis::BasisFunction,
    hf::mo::MolecularOrbitals,
    integrals::{DefaultIntegrator, ElectronTensor, Integrator},
};

use super::{utils, HartreeFockInput};

/// The output of a restricted hartree fock calculation
#[derive(Debug)]
#[non_exhaustive]
#[allow(unused)]
pub struct UnrestrictedHartreeFockOutput {
    /// The spin up molecular orbitals that were found in the hartree fock calculation.
    /// These are sorted by ascending order in energy.
    pub(crate) orbitals_alpha: MolecularOrbitals,
    /// The spin down molecular orbitals that were found in the hartree fock calculation.
    /// These are sorted by ascending order in energy.
    pub(crate) orbitals_beta: MolecularOrbitals,
    /// the basis that was used in the hartree fock calculation. This is necessary
    /// to be able to for example evaluate the molecular orbitals that were found
    pub(crate) basis: Vec<BasisFunction>,
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
    input: &HartreeFockInput,
) -> Option<UnrestrictedHartreeFockOutput> {
    // exchangable integrator
    let integrator = DefaultIntegrator::default();

    // TODO: group integral terms by similar terms?
    let basis = input
        .molecule
        .atoms
        .iter()
        .flat_map(|atom| {
            let atomic_basis = input
                .basis_set
                .for_atom(atom)
                .unwrap_or_else(|| panic!("no basis for element {:?}", atom.element_type));

            atomic_basis
                .basis_functions()
                .map(|function_type| BasisFunction {
                    contracted_gaussian: function_type.clone(),
                    position: atom.position,
                })
        })
        .collect::<Vec<_>>();

    let n_basis = basis.len();

    let n_alpha = input.n_alpha();
    let n_beta = input.n_beta();

    let nuclear_repulsion = compute_nuclear_repulsion(&input.molecule.atoms);
    log::debug!("nulcear repulsion energy: {nuclear_repulsion}");

    // TODO: do we need to pre-calculate all of the integrals? I don't think, for example, all ERI integrals are actually used.
    //  if we could skip some of them, that would be a huge performance gain.
    let overlap = compute_overlap_matrix(&basis, &integrator);
    log::debug!("overlap matrix: {overlap:0.4}");
    let kinetic = compute_kinetic_matrix(&basis, &integrator);
    log::debug!("kinetic matrix: {overlap:0.4}");
    let nuclear = compute_nuclear_matrix(&basis, &input.molecule.atoms, &integrator);
    log::debug!("nuclear matrix: {overlap:0.4}");
    let electron = ElectronTensor::from_basis(&basis, &integrator);

    let core_hamiltonian = kinetic + nuclear;
    let transform = compute_transformation_matrix(&overlap);

    let mut density_alpha = DMatrix::zeros(n_basis, n_basis);
    let mut density_beta = DMatrix::zeros(n_basis, n_basis);

    // TODO: use these again
    let mut electron_terms = vec![0.0; n_basis.pow(4)];
    for j in 0..n_basis {
        for i in 0..n_basis {
            for x in 0..n_basis {
                for y in 0..n_basis {
                    electron_terms[j * n_basis.pow(3) + i * n_basis.pow(2) + y * n_basis + x] =
                        electron[(i, j, x, y)] - 0.5 * electron[(i, x, j, y)];
                }
            }
        }
    }

    let mut electronic_hamiltonians = vec![DMatrix::zeros(n_basis, n_basis); 2];
    let mut orbital_energies = vec![DVector::zeros(n_basis); 2];
    let mut coefficient_matrices = vec![DMatrix::zeros(n_basis, n_basis); 2];

    // start of scf iteration
    for iteration in 0..=input.max_iterations {
        for spin in 0..=1 {
            // "main" density and "other" density
            let (density_one, density_two) = match spin {
                0 => (&mut density_alpha, &density_beta),
                1 => (&mut density_beta, &density_alpha),
                _ => unreachable!(),
            };

            let electronic_hamiltonian =
                compute_electronic_hamiltonian(&density_one, &density_two, &electron, n_basis);

            let fock = &core_hamiltonian + &electronic_hamiltonian;
            electronic_hamiltonians[spin] = electronic_hamiltonian;

            let transformed_fock = &transform.transpose() * (&fock * &transform);
            let (transformed_coefficients, spin_orbital_energies) =
                utils::sorted_eigs(transformed_fock);
            let coefficients = &transform * &transformed_coefficients;

            coefficient_matrices[spin] = coefficients;
            orbital_energies[spin] = spin_orbital_energies;
        }

        let mut density_rms = 0.0;
        // second loop, because we need the new coefficients to compute the new density matrices
        for spin in 0..=1 {
            // "main" density and "other" density
            let (old_density, coefficients_a, coefficients_b, electrons_a, electrons_b) = match spin
            {
                0 => (
                    &mut density_alpha,
                    &coefficient_matrices[0],
                    &coefficient_matrices[1],
                    n_alpha,
                    n_beta,
                ),
                1 => (
                    &mut density_beta,
                    &coefficient_matrices[1],
                    &coefficient_matrices[0],
                    n_beta,
                    n_alpha,
                ),
                _ => unreachable!(),
            };

            let new_density = compute_updated_density(
                coefficients_a,
                coefficients_b,
                n_basis,
                electrons_a,
                electrons_b,
            );

            const F: f64 = 0.5;
            let density_change = &new_density - &*old_density;
            *old_density += &density_change * F;

            let self_rms =
                (density_change.map_diagonal(|entry| entry.powi(2)).sum() / n_basis as f64).sqrt();

            density_rms += self_rms;

            log::info!(
                "spin {} - density rms {self_rms:03.3e}",
                ["up", "down"][spin]
            )
        }

        if density_rms / 2.0 < input.epsilon {
            let electronic_energy = 0.5
                * ((&density_alpha * &core_hamiltonian).trace()
                    + (&density_beta * &core_hamiltonian).trace()
                    + 0.5 * (&density_alpha * &electronic_hamiltonians[0]).trace()
                    + 0.5 * (&density_beta * &electronic_hamiltonians[1]).trace());

            println!(
                "{:0.6?} | {:0.6?}",
                orbital_energies[0].as_slice(),
                orbital_energies[1].as_slice()
            );

            return Some(UnrestrictedHartreeFockOutput {
                orbitals_alpha: MolecularOrbitals::from_matrix(&coefficient_matrices[0]),
                orbitals_beta: MolecularOrbitals::from_matrix(&coefficient_matrices[1]),
                basis,
                orbital_energies_alpha: orbital_energies[0].as_slice().to_vec(),
                orbital_energies_beta: orbital_energies[1].as_slice().to_vec(),
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
            potential += (atoms[atom_a].nuclear_charge() * atoms[atom_b].nuclear_charge()) as f64
                / (atoms[atom_b].position - atoms[atom_a].position).norm()
        }
    }
    potential
}

pub fn compute_overlap_matrix(
    basis: &[BasisFunction],
    integrator: &impl Integrator<Item = BasisFunction>,
) -> DMatrix<f64> {
    utils::symmetric_matrix(basis.len(), |i, j| {
        let overlap_ij = integrator.overlap((&basis[i], &basis[j]));
        log::trace!("overlap ({i}{j}) = {overlap_ij}");
        overlap_ij
    })
}

pub fn compute_kinetic_matrix(
    basis: &[BasisFunction],
    integrator: &impl Integrator<Item = BasisFunction>,
) -> DMatrix<f64> {
    utils::symmetric_matrix(basis.len(), |i, j| {
        let kinetic_ij = integrator.kinetic((&basis[i], &basis[j]));
        log::trace!("kinetic ({i}{j}) = {kinetic_ij}");
        kinetic_ij
    })
}

pub fn compute_nuclear_matrix(
    basis: &[BasisFunction],
    nuclei: &[Atom],
    integrator: &impl Integrator<Item = BasisFunction>,
) -> DMatrix<f64> {
    utils::symmetric_matrix(basis.len(), |i, j| {
        let nuclear_ij = integrator.nuclear((&basis[i], &basis[j]), nuclei);
        log::trace!("nuclear ({i}{j}) = {nuclear_ij}");
        nuclear_ij
    })
}

fn compute_transformation_matrix(overlap: &DMatrix<f64>) -> DMatrix<f64> {
    let (u, _) = utils::eigs(overlap.clone());
    let diagonal_matrix = &u.transpose() * (overlap * &u);

    let diagonal_inv_sqrt =
        DMatrix::from_diagonal(&diagonal_matrix.map_diagonal(|f| f.sqrt().recip()));
    &u * (diagonal_inv_sqrt * &u.transpose())
}

fn compute_electronic_hamiltonian(
    density_one: &DMatrix<f64>,
    density_two: &DMatrix<f64>,
    multi: &ElectronTensor,
    n_basis: usize,
) -> DMatrix<f64> {
    utils::symmetric_matrix(n_basis, |i, j| {
        let mut sum = 0.0;
        for y in 0..n_basis {
            for x in 0..n_basis {
                sum += 0.5 * density_one[(x, y)] * multi[(i, j, x, y)]
                    + 0.5 * density_two[(x, y)] * multi[(i, j, x, y)]
                    - density_one[(x, y)] * multi[(i, x, j, y)];
            }
        }
        sum
    })
}

fn compute_updated_density(
    coefficients_one: &DMatrix<f64>,
    coefficients_two: &DMatrix<f64>,
    n_basis: usize,
    n_electrons_one: usize,
    n_electrons_two: usize,
) -> DMatrix<f64> {
    utils::symmetric_matrix(n_basis, |i, j| {
        let mut sum = 0.0;
        for k in 0..n_electrons_one {
            sum += coefficients_one[(i, k)] * coefficients_one[(j, k)];
        }
        for k in 0..n_electrons_two {
            sum += coefficients_two[(i, k)] * coefficients_two[(j, k)];
        }
        sum
    })
}
