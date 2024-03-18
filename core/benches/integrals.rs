use core::{
    basis::BasisFunction,
    hf::scf,
    integrals::{mmd, Integrator},
    testing::TestInstance,
};
use std::error::Error;

use criterion::{criterion_group, criterion_main, Criterion};

macro_rules! test {
    ($mol:literal, $basis:literal) => {{
        ::serde_json::from_reader::<_, TestInstance>(::std::fs::File::open(format!(
            "benches/test-inputs/{}_{}.json",
            $mol, $basis
        ))?)?
    }};
}

fn bench_kinetic(
    c: &mut Criterion,
    integrator: &impl Integrator<Function = BasisFunction>,
) -> Result<(), Box<dyn Error>> {
    let hydrogen_sto3g = test!("hydrogen", "STO-3G");
    let hydrogen_631g = test!("hydrogen", "6-31G");

    let water_sto3g = test!("water", "STO-3G");
    let water_631g = test!("water", "6-31G");

    let benzene_sto3g = test!("benzene", "STO-3G");
    let benzene_631g = test!("benzene", "6-31G");

    let mut bench_instance = move |name: &str, instance: &TestInstance| {
        let basis_functions = instance.basis_functions();

        c.bench_function(name, move |b| {
            b.iter(move || scf::compute_kinetic_matrix(basis_functions, integrator))
        });
    };
    bench_instance("Hydrogen STO-3G", &hydrogen_sto3g);
    bench_instance("Hydrogen 6-31G", &hydrogen_631g);

    bench_instance("Water STO-3G", &water_sto3g);
    bench_instance("Water 6-31G", &water_631g);

    bench_instance("Benzene STO-3G", &benzene_sto3g);
    bench_instance("Benzene 6-31G", &benzene_631g);

    Ok(())
}

fn bench_integrals(c: &mut Criterion) {
    bench_kinetic(c, &mmd::McMurchieDavidson::default()).unwrap();
}

criterion_group!(benches, bench_integrals);
criterion_main!(benches);
