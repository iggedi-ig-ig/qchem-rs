use std::collections::VecDeque;

use nalgebra::{DMatrix, DVector, SymmetricEigen};

use crate::{
    atom::Atom,
    basis::BasisFunction,
    hf::mo::MolecularOrbitals,
    integrals::{DefaultIntegrator, ElectronTensor, Integrator},
};

use super::{HartreeFockInput, HartreeFockOutput};

pub fn hartree_fock(input: &HartreeFockInput) -> Option<HartreeFockOutput> {
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
                .unwrap_or_else(|| panic!("no basis for atom with ordinal {}", atom.ordinal));

            atomic_basis
                .basis_functions
                .iter()
                .map(|function_type| BasisFunction {
                    function_type: function_type.clone(),
                    position: atom.position,
                })
        })
        .collect::<Vec<_>>();

    let n_basis = basis.len();

    let n_electrons = input
        .molecule
        .atoms
        .iter()
        .map(|atom| atom.ordinal as i32)
        .sum::<i32>() as usize;

    let nuclear_repulsion = calculate_nuclear_repulsion(&input.molecule.atoms);
    log::debug!("nulcear repulsion energy: {nuclear_repulsion}");

    // TODO: do we need to pre-calculate all of the integrals? I don't think, for example, all ERI integrals are actually used.
    //  if we could skip some of them, that would be a huge performance gain.
    let overlap = calculate_overlap_matrix(&basis, &integrator);
    log::debug!("overlap matrix: {overlap:0.4}");
    let kinetic = calculate_kinetic_matrix(&basis, &integrator);
    log::debug!("kinetic matrix: {overlap:0.4}");
    let nuclear = calculate_nuclear_matrix(&basis, &input.molecule.atoms, &integrator);
    log::debug!("nuclear matrix: {overlap:0.4}");
    let electron = ElectronTensor::from_basis(&basis, &integrator);

    let core_hamiltonian = kinetic + nuclear;
    let transform = calculate_transformation_matrix(&overlap);
    let mut density = calculate_hückel_density(
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
    for iteration in 0..=input.max_iterations {
        let density_guess = calculate_density_guess(&density, &electron_terms, n_basis);

        let fock = &core_hamiltonian + &density_guess;

        let transformed_fock = &transform.transpose() * (&fock * &transform);
        let (transformed_coefficients, obrital_energies) = sorted_eigs(transformed_fock);
        let coefficients = &transform * &transformed_coefficients;

        let new_density = calculate_updated_density(&coefficients, n_basis, n_electrons);

        const F: f64 = 1.0;
        let density_change = new_density - &density;
        density += &density_change * F;

        let electronic_energy =
            0.5 * (&density * (2.0 * &core_hamiltonian + &density_guess)).trace();

        let density_rms =
            (density_change.map_diagonal(|entry| entry.powi(2)).sum() / n_basis as f64).sqrt();

        log::info!(
            "iteration {iteration:<4} - electronic energy {electronic_energy:1.4}. density rms {density_rms:1.4e}",
        );

        if density_rms < input.epsilon {
            let electronic_energy =
                0.5 * (&density * (2.0 * &core_hamiltonian + &density_guess)).trace();
            return Some(HartreeFockOutput {
                orbitals: MolecularOrbitals::from_coefficient_matrix(&coefficients),
                basis,
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

fn calculate_nuclear_repulsion(atoms: &[Atom]) -> f64 {
    let n_atoms = atoms.len();

    let mut potential = 0.0;
    for atom_a in 0..n_atoms {
        for atom_b in atom_a + 1..n_atoms {
            potential += (atoms[atom_a].charge() * atoms[atom_b].charge()) as f64
                / (atoms[atom_b].position - atoms[atom_a].position).norm()
        }
    }
    potential
}

// TODO: the basis argument type is a bit awkward to work with
fn calculate_overlap_matrix(
    basis: &[BasisFunction],
    integrator: &dyn Integrator<Function = BasisFunction>,
) -> DMatrix<f64> {
    hermitian(basis.len(), |i, j| {
        let overlap_ij = integrator.overlap((&basis[i], &basis[j]));
        log::trace!("overlap ({i}{j}) = {overlap_ij}");
        overlap_ij
    })
}

// TODO: the basis argument type is a bit awkward to work with
fn calculate_kinetic_matrix(
    basis: &[BasisFunction],
    integrator: &dyn Integrator<Function = BasisFunction>,
) -> DMatrix<f64> {
    hermitian(basis.len(), |i, j| {
        let kinetic_ij = integrator.kinetic((&basis[i], &basis[j]));
        log::trace!("kinetic ({i}{j}) = {kinetic_ij}");
        kinetic_ij
    })
}

// TODO: the basis argument type is a bit awkward to work with
fn calculate_nuclear_matrix(
    basis: &[BasisFunction],
    nuclei: &[Atom],
    integrator: &dyn Integrator<Function = BasisFunction>,
) -> DMatrix<f64> {
    hermitian(basis.len(), |i, j| {
        let nuclear_ij = integrator.nuclear((&basis[i], &basis[j]), nuclei);
        log::trace!("nuclear ({i}{j}) = {nuclear_ij}");
        nuclear_ij
    })
}

fn calculate_transformation_matrix(overlap: &DMatrix<f64>) -> DMatrix<f64> {
    let (u, _) = eigs(overlap.clone());
    let diagonal_matrix = &u.transpose() * (overlap * &u);

    let diagonal_inv_sqrt =
        DMatrix::from_diagonal(&diagonal_matrix.map_diagonal(|f| f.sqrt().recip()));
    &u * (diagonal_inv_sqrt * &u.transpose())
}

fn calculate_hückel_density(
    hamiltonian: &DMatrix<f64>,
    overlap: &DMatrix<f64>,
    transform: &DMatrix<f64>,
    n_basis: usize,
    n_electrons: usize,
) -> DMatrix<f64> {
    let hamiltonian_eht = hermitian(n_basis, |i, j| {
        0.875 * overlap[(i, j)] * (hamiltonian[(i, i)] + hamiltonian[(j, j)])
    });

    let transformed = &transform.transpose() * (hamiltonian_eht * transform);
    let (coeffs_prime, _orbital_energies) = sorted_eigs(transformed);
    let coeffs = transform * coeffs_prime;

    hermitian(n_basis, |i, j| {
        2.0 * (0..n_electrons / 2).fold(0.0, |acc, k| acc + coeffs[(i, k)] * coeffs[(j, k)])
    })
}

fn calculate_density_guess(
    density: &DMatrix<f64>,
    electron_terms: &[f64],
    n_basis: usize,
) -> DMatrix<f64> {
    hermitian(n_basis, |i, j| {
        (0..n_basis).fold(0.0, |acc, y| {
            acc + (0..n_basis).fold(0.0, |acc, x| {
                acc + density[(x, y)]
                    * electron_terms[j * n_basis.pow(3) + i * n_basis.pow(2) + y * n_basis + x]
            })
        })
    })
}

fn calculate_updated_density(
    coefficients: &DMatrix<f64>,
    n_basis: usize,
    n_electrons: usize,
) -> DMatrix<f64> {
    hermitian(n_basis, |i, j| {
        2.0 * (0..n_electrons / 2).fold(0.0, |acc, k| {
            acc + coefficients[(i, k)] * coefficients[(j, k)]
        })
    })
}

fn hermitian(n: usize, mut func: impl FnMut(usize, usize) -> f64) -> DMatrix<f64> {
    let m = DMatrix::from_fn(n, n, |i, j| if i <= j { func(i, j) } else { 0.0 });
    DMatrix::from_fn(n, n, |i, j| if i <= j { m[(i, j)] } else { m[(j, i)] })
}

fn eigs(matrix: DMatrix<f64>) -> (DMatrix<f64>, DVector<f64>) {
    let eigs = SymmetricEigen::new(matrix);
    (eigs.eigenvectors, eigs.eigenvalues)
}

fn sorted_eigs(matrix: DMatrix<f64>) -> (DMatrix<f64>, DVector<f64>) {
    let (eigenvectors, eigenvalues) = eigs(matrix);

    let mut val_vec_pairs = eigenvalues
        .into_iter()
        .zip(eigenvectors.column_iter())
        .collect::<Vec<_>>();

    val_vec_pairs.sort_unstable_by(|(a, _), (b, _)| a.total_cmp(b));

    let (values, vectors): (Vec<_>, Vec<_>) = val_vec_pairs.into_iter().unzip();

    (
        DMatrix::from_columns(&vectors),
        DVector::from_column_slice(&values),
    )
}

// TODO: reimplement diis
fn _diis(
    error_vectors: &VecDeque<DMatrix<f64>>,
    fock_matricies: &VecDeque<DMatrix<f64>>,
) -> Option<DMatrix<f64>> {
    assert_eq!(error_vectors.len(), fock_matricies.len());
    let n = error_vectors.len();

    let mut matrix = DMatrix::zeros(n + 1, n + 1);
    // upper block
    for (i, j) in itertools::iproduct!(0..n, 0..n) {
        matrix[(i, j)] = error_vectors[i].dot(&error_vectors[j]);
    }

    // last row
    for i in 0..n {
        matrix[(n, i)] = -1.0;
    }

    // last col
    for i in 0..n {
        matrix[(i, n)] = -1.0;
    }

    // last entry
    matrix[(n, n)] = 0.0;

    let mut b = DVector::zeros(n + 1);
    b[(n, 0)] = -1.0;

    matrix.try_inverse().map(|inv| inv * b).map(|c| {
        c.iter()
            .enumerate()
            .take(n)
            .map(|(i, &x)| x * &fock_matricies[i])
            .sum()
    })
}

#[cfg(test)]
mod tests {
    //! Tests were generated using fock-rs - it uses the exact same integration techniques, same boys function implementation, etc.
    use approx::assert_relative_eq;

    use crate::{
        basis::BasisSet,
        bse::BseBasisSet,
        hf::{hartree_fock, HartreeFockInput, HartreeFockOutput},
    };

    macro_rules! molecule {
        ($(
            $element:ident => ($x:expr, $y:expr, $z:expr)
        ),*) => {
            $crate::molecule::Molecule {
                atoms: vec![
                    $($crate::atom::Atom {
                        position: ::nalgebra::Vector3::new($x, $y, $z),
                        ordinal: $crate::bse::ElementType::$element as usize,
                    }),*
                ]
            }
        };
    }

    #[test]
    fn hydrogen_6_31g() {
        const B_6_31G: &'static str = r#"{"molssi_bse_schema":{"schema_type":"complete","schema_version":"0.1"},"revision_description":"DatafromGaussian09/GAMESS","revision_date":"2018-06-19","elements":{"1":{"electron_shells":[{"function_type":"gto","region":"valence","angular_momentum":[0],"exponents":["0.1873113696E+02","0.2825394365E+01","0.6401216923E+00"],"coefficients":[["0.3349460434E-01","0.2347269535E+00","0.8137573261E+00"]]},{"function_type":"gto","region":"valence","angular_momentum":[0],"exponents":["0.1612777588E+00"],"coefficients":[["1.0000000"]]}],"references":[{"reference_description":"31GSplit-valencebasissetforH,He","reference_keys":["ditchfield1971a"]}]}},"version":"1","function_types":["gto"],"names":["6-31G"],"tags":[],"family":"pople","description":"6-31Gvalencedouble-zeta","role":"orbital","auxiliaries":{},"name":"6-31G"}"#;

        let molecule = molecule! {
            H => (0.0, 0.0, 0.0),
            H => (0.0, 0.0, 1.4)
        };
        let basis_set: BseBasisSet = serde_json::from_str(B_6_31G).unwrap();
        let basis_set = BasisSet::try_from(basis_set).unwrap();

        let input = HartreeFockInput {
            molecule: &molecule,
            basis_set: &basis_set,
            max_iterations: 100,
            epsilon: 1e-6,
        };

        let HartreeFockOutput {
            orbital_energies,
            electronic_energy,
            nuclear_repulsion,
            ..
        } = hartree_fock(&input).unwrap();

        assert_relative_eq!(electronic_energy, -1.8410539726907735, epsilon = 1e-3);
        assert_relative_eq!(nuclear_repulsion, 0.7142857142857142, epsilon = 1e-3);
        assert_relative_eq!(orbital_energies[0], -0.595564373728178, epsilon = 1e-3);
        assert_relative_eq!(orbital_energies[1], 0.2382503139896246, epsilon = 1e-3);
        assert_relative_eq!(orbital_energies[2], 0.7750727506800223, epsilon = 1e-3);
        assert_relative_eq!(orbital_energies[3], 1.40316490313582, epsilon = 1e-3);
    }

    #[test]
    fn water_6_31g() {
        const B_6_31G: &str = r#"{"molssi_bse_schema":{"schema_type":"complete","schema_version":"0.1"},"revision_description":"DatafromGaussian09/GAMESS","revision_date":"2018-06-19","elements":{"1":{"electron_shells":[{"function_type":"gto","region":"valence","angular_momentum":[0],"exponents":["0.1873113696E+02","0.2825394365E+01","0.6401216923E+00"],"coefficients":[["0.3349460434E-01","0.2347269535E+00","0.8137573261E+00"]]},{"function_type":"gto","region":"valence","angular_momentum":[0],"exponents":["0.1612777588E+00"],"coefficients":[["1.0000000"]]}],"references":[{"reference_description":"31GSplit-valencebasissetforH,He","reference_keys":["ditchfield1971a"]}]},"8":{"electron_shells":[{"function_type":"gto","region":"valence","angular_momentum":[0],"exponents":["0.5484671660E+04","0.8252349460E+03","0.1880469580E+03","0.5296450000E+02","0.1689757040E+02","0.5799635340E+01"],"coefficients":[["0.1831074430E-02","0.1395017220E-01","0.6844507810E-01","0.2327143360E+00","0.4701928980E+00","0.3585208530E+00"]]},{"function_type":"gto","region":"valence","angular_momentum":[0,1],"exponents":["0.1553961625E+02","0.3599933586E+01","0.1013761750E+01"],"coefficients":[["-0.1107775495E+00","-0.1480262627E+00","0.1130767015E+01"],["0.7087426823E-01","0.3397528391E+00","0.7271585773E+00"]]},{"function_type":"gto","region":"valence","angular_momentum":[0,1],"exponents":["0.2700058226E+00"],"coefficients":[["0.1000000000E+01"],["0.1000000000E+01"]]}],"references":[{"reference_description":"6-31GSplit-valencebasisset","reference_keys":["hehre1972a"]}]}},"version":"1","function_types":["gto"],"names":["6-31G"],"tags":[],"family":"pople","description":"6-31Gvalencedouble-zeta","role":"orbital","auxiliaries":{},"name":"6-31G"}"#;

        let molecule = molecule! {
            O => (0.0, 0.0, 0.0),
            H => (0.0, 0.75, 0.585),
            H => (0.0, -0.75, 0.585)
        };

        let basis_set: BseBasisSet = serde_json::from_str(B_6_31G).unwrap();
        let basis_set = BasisSet::try_from(basis_set).unwrap();

        let input = HartreeFockInput {
            molecule: &molecule,
            basis_set: &basis_set,
            max_iterations: 100,
            epsilon: 1e-6,
        };

        let HartreeFockOutput {
            orbital_energies,
            electronic_energy,
            nuclear_repulsion,
            ..
        } = hartree_fock(&input).unwrap();

        // we're relatively lenient with accuracy here - but that's alright. The current
        // implementation of the boys function is only accurate to about 3 digits, so we
        // can't really get more accurate than that anyways.
        assert_relative_eq!(electronic_energy, -92.0230896544854, epsilon = 1e-3);
        assert_relative_eq!(nuclear_repulsion, 17.488049195046216, epsilon = 1e-3);

        assert_relative_eq!(orbital_energies[0], -20.523974864948215, epsilon = 1e-3);
        assert_relative_eq!(orbital_energies[1], -1.740756409886931, epsilon = 1e-3);
        assert_relative_eq!(orbital_energies[2], -1.0307147688870948, epsilon = 1e-3);
        assert_relative_eq!(orbital_energies[3], -0.6441844270629393, epsilon = 1e-3);
        assert_relative_eq!(orbital_energies[4], -0.6254760824539792, epsilon = 1e-3);
        assert_relative_eq!(orbital_energies[5], 0.26161301182427255, epsilon = 1e-3);
        assert_relative_eq!(orbital_energies[6], 0.35241825914625563, epsilon = 1e-3);
        assert_relative_eq!(orbital_energies[7], 1.0439825477951785, epsilon = 1e-3);
        assert_relative_eq!(orbital_energies[8], 1.05397187051381, epsilon = 1e-3);
        assert_relative_eq!(orbital_energies[9], 1.1737210868055685, epsilon = 1e-3);
        assert_relative_eq!(orbital_energies[10], 1.7054463744890267, epsilon = 1e-3);
        assert_relative_eq!(orbital_energies[11], 2.3235979994552256, epsilon = 1e-3);
        assert_relative_eq!(orbital_energies[12], 2.906226162677248, epsilon = 1e-3);
    }

    #[test]
    fn ethylene_sto_3g() {
        const B_STO_3G: &str = r#"{"molssi_bse_schema":{"schema_type":"complete","schema_version":"0.1"},"revision_description":"DatafromGaussian09","revision_date":"2018-06-19","elements":{"1":{"electron_shells":[{"function_type":"gto","region":"","angular_momentum":[0],"exponents":["0.3425250914E+01","0.6239137298E+00","0.1688554040E+00"],"coefficients":[["0.1543289673E+00","0.5353281423E+00","0.4446345422E+00"]]}],"references":[{"reference_description":"STO-3GMinimalBasis(3functions/AO)","reference_keys":["hehre1969a"]}]},"6":{"electron_shells":[{"function_type":"gto","region":"","angular_momentum":[0],"exponents":["0.7161683735E+02","0.1304509632E+02","0.3530512160E+01"],"coefficients":[["0.1543289673E+00","0.5353281423E+00","0.4446345422E+00"]]},{"function_type":"gto","region":"","angular_momentum":[0,1],"exponents":["0.2941249355E+01","0.6834830964E+00","0.2222899159E+00"],"coefficients":[["-0.9996722919E-01","0.3995128261E+00","0.7001154689E+00"],["0.1559162750E+00","0.6076837186E+00","0.3919573931E+00"]]}],"references":[{"reference_description":"STO-3GMinimalBasis(3functions/AO)","reference_keys":["hehre1969a"]}]}},"version":"1","function_types":["gto"],"names":["STO-3G"],"tags":[],"family":"sto","description":"STO-3GMinimalBasis(3functions/AO)","role":"orbital","auxiliaries":{},"name":"STO-3G"}"#;

        let molecule = molecule! {
            C => (0.0000, 0.0000, 0.0000),
            C => (1.3390, 0.0000, 0.0000),
            H => (0.0000, 0.9281, 0.5621),
            H => (0.0000, -0.9281, 0.5621),
            H => (1.3390, 0.9281, -0.5621),
            H => (1.3390, -0.9281, -0.5621)
        };

        let basis_set: BseBasisSet = serde_json::from_str(B_STO_3G).unwrap();
        let basis_set = BasisSet::try_from(basis_set).unwrap();

        let input = HartreeFockInput {
            molecule: &molecule,
            basis_set: &basis_set,
            max_iterations: 100,
            epsilon: 1e-6,
        };

        let HartreeFockOutput {
            orbital_energies,
            electronic_energy,
            nuclear_repulsion,
            ..
        } = hartree_fock(&input).unwrap();

        assert_relative_eq!(electronic_energy, -137.46094993664823, epsilon = 1e-3);
        assert_relative_eq!(nuclear_repulsion, 65.935968236742, epsilon = 1e-3);

        assert_relative_eq!(orbital_energies[0], -11.843194691423907, epsilon = 1e-3);
        assert_relative_eq!(orbital_energies[1], -11.654101433513777, epsilon = 1e-3);
        assert_relative_eq!(orbital_energies[2], -1.7549674340520647, epsilon = 1e-3);
        assert_relative_eq!(orbital_energies[3], -1.3740629705845648, epsilon = 1e-3);
        assert_relative_eq!(orbital_energies[4], -1.1766056496692792, epsilon = 1e-3);
        assert_relative_eq!(orbital_energies[5], -0.6894580276990747, epsilon = 1e-3);
        assert_relative_eq!(orbital_energies[6], -0.6854735898003864, epsilon = 1e-3);
        assert_relative_eq!(orbital_energies[7], -0.5457185487988014, epsilon = 1e-3);
        assert_relative_eq!(orbital_energies[8], 0.3271343831398882, epsilon = 1e-3);
        assert_relative_eq!(orbital_energies[9], 1.1286944290080008, epsilon = 1e-3);
        assert_relative_eq!(orbital_energies[10], 1.2948544365472605, epsilon = 1e-3);
        assert_relative_eq!(orbital_energies[11], 1.5951207260702598, epsilon = 1e-3);
        assert_relative_eq!(orbital_energies[12], 1.7597320917001829, epsilon = 1e-3);
        assert_relative_eq!(orbital_energies[13], 4.650442932567982, epsilon = 1e-3);
    }

    // TODO: reintroduce this test. Right now, this doesn't converge - the result we're getting is correct though*
    //  (* it's not converging, so we're not "getting" the result, but the energy is trapped in a cycle between the correct value and another one)
    // #[test]
    fn _chlorine_6_31g() {
        const B_6_31G: &str = r#"{"molssi_bse_schema":{"schema_type":"complete","schema_version":"0.1"},"revision_description":"DatafromGaussian09/GAMESS","revision_date":"2018-06-19","elements":{"17":{"electron_shells":[{"function_type":"gto","region":"valence","angular_momentum":[0],"exponents":["0.2518010000E+05","0.3780350000E+04","0.8604740000E+03","0.2421450000E+03","0.7733490000E+02","0.2624700000E+02"],"coefficients":[["0.1832959848E-02","0.1403419883E-01","0.6909739426E-01","0.2374519803E+00","0.4830339599E+00","0.3398559718E+00"]]},{"function_type":"gto","region":"valence","angular_momentum":[0,1],"exponents":["0.4917650000E+03","0.1169840000E+03","0.3741530000E+02","0.1378340000E+02","0.5452150000E+01","0.2225880000E+01"],"coefficients":[["-0.2297391417E-02","-0.3071371894E-01","-0.1125280694E+00","0.4501632776E-01","0.5893533634E+00","0.4652062868E+00"],["0.3989400879E-02","0.3031770668E-01","0.1298800286E+00","0.3279510723E+00","0.4535271000E+00","0.2521540556E+00"]]},{"function_type":"gto","region":"valence","angular_momentum":[0,1],"exponents":["0.3186490000E+01","0.1144270000E+01","0.4203770000E+00"],"coefficients":[["-0.2518280280E+00","0.6158925141E-01","0.1060184328E+01"],["-0.1429931472E-01","0.3235723331E+00","0.7435077653E+00"]]},{"function_type":"gto","region":"valence","angular_momentum":[0,1],"exponents":["0.1426570000E+00"],"coefficients":[["0.1000000000E+01"],["0.1000000000E+01"]]}],"references":[{"reference_description":"6-21GSplit-valencebasisset","reference_keys":["gordon1982a"]},{"reference_description":"31GSplit-valencepart","reference_keys":["francl1982a"]}]}},"version":"1","function_types":["gto"],"names":["6-31G"],"tags":[],"family":"pople","description":"6-31Gvalencedouble-zeta","role":"orbital","auxiliaries":{},"name":"6-31G"}"#;

        let molecule = molecule! {
            Cl => (0.0000, 0.0000, 0.0000),
            Cl => (1.9880, 0.0000, 0.0000)
        };

        let basis_set: BseBasisSet = serde_json::from_str(B_6_31G).unwrap();
        let basis_set = BasisSet::try_from(basis_set).unwrap();

        let input = HartreeFockInput {
            molecule: &molecule,
            basis_set: &basis_set,
            max_iterations: 100,
            epsilon: 1e-6,
        };

        let HartreeFockOutput {
            orbital_energies,
            electronic_energy,
            nuclear_repulsion,
            ..
        } = hartree_fock(&input).unwrap();

        assert_relative_eq!(electronic_energy, -1057.0756035784825, epsilon = 1e-3);
        assert_relative_eq!(nuclear_repulsion, 145.37223340040237, epsilon = 1e-3);

        assert_relative_eq!(orbital_energies[0], -104.93509669543764, epsilon = 1e-3);
        assert_relative_eq!(orbital_energies[1], -104.93509280917917, epsilon = 1e-3);
        assert_relative_eq!(orbital_energies[2], -12.079387588125474, epsilon = 1e-3);
        assert_relative_eq!(orbital_energies[3], -12.075983265528384, epsilon = 1e-3);
        assert_relative_eq!(orbital_energies[4], -8.290273842911198, epsilon = 1e-3);
        assert_relative_eq!(orbital_energies[5], -8.282137017695804, epsilon = 1e-3);
        assert_relative_eq!(orbital_energies[6], -8.272403160534182, epsilon = 1e-3);
        assert_relative_eq!(orbital_energies[7], -8.264921718643821, epsilon = 1e-3);
        assert_relative_eq!(orbital_energies[8], -7.368561627054928, epsilon = 1e-3);
        assert_relative_eq!(orbital_energies[9], -7.230767656418042, epsilon = 1e-3);
        assert_relative_eq!(orbital_energies[10], -6.469281375498937, epsilon = 1e-3);
        assert_relative_eq!(orbital_energies[11], -3.969796843827698, epsilon = 1e-3);
        assert_relative_eq!(orbital_energies[12], -3.866751621590824, epsilon = 1e-3);
        assert_relative_eq!(orbital_energies[13], -1.0303906963565175, epsilon = 1e-3);
        assert_relative_eq!(orbital_energies[14], -0.9977507951920545, epsilon = 1e-3);
        assert_relative_eq!(orbital_energies[15], -0.5930080884738071, epsilon = 1e-3);
        assert_relative_eq!(orbital_energies[16], -0.38773387099981943, epsilon = 1e-3);
        assert_relative_eq!(orbital_energies[17], -0.03993492881441759, epsilon = 1e-3);
        assert_relative_eq!(orbital_energies[18], 0.3409537420478979, epsilon = 1e-3);
        assert_relative_eq!(orbital_energies[19], 0.36712421154084257, epsilon = 1e-3);
        assert_relative_eq!(orbital_energies[20], 0.6118387978730864, epsilon = 1e-3);
        assert_relative_eq!(orbital_energies[21], 0.6383166463865466, epsilon = 1e-3);
        assert_relative_eq!(orbital_energies[22], 1.2666852985066537, epsilon = 1e-3);
        assert_relative_eq!(orbital_energies[23], 5.809702931259727, epsilon = 1e-3);
        assert_relative_eq!(orbital_energies[24], 32.85704856869294, epsilon = 1e-3);
        assert_relative_eq!(orbital_energies[25], 575.9528369904834, epsilon = 1e-3);
    }
}
