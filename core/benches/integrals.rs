use core::{
    hf::scf,
    integrals::{electron_tensor::ElectronTensor, mmd::McMurchieDavidson},
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

const INTEGRATOR: McMurchieDavidson = McMurchieDavidson;

fn bench_overlap(c: &mut Criterion, instances: &[&TestInstance]) -> Result<(), Box<dyn Error>> {
    for instance in instances {
        let basis_functions = instance.basis_functions();

        c.bench_function(&format!("Overlap {}", instance.name), move |b| {
            b.iter(move || scf::compute_overlap_matrix(basis_functions, &INTEGRATOR))
        });
    }

    Ok(())
}

fn bench_kinetic(c: &mut Criterion, instances: &[&TestInstance]) -> Result<(), Box<dyn Error>> {
    for instance in instances {
        let basis_functions = instance.basis_functions();

        c.bench_function(&format!("Kinetic {}", instance.name), move |b| {
            b.iter(move || scf::compute_kinetic_matrix(basis_functions, &INTEGRATOR))
        });
    }

    Ok(())
}

fn bench_electron(c: &mut Criterion, instances: &[&TestInstance]) -> Result<(), Box<dyn Error>> {
    for instance in instances {
        let basis_functions = instance.basis_functions();

        c.bench_function(&format!("Electron Repulsion {}", instance.name), move |b| {
            b.iter(move || ElectronTensor::from_basis(basis_functions, &INTEGRATOR))
        });
    }

    Ok(())
}

fn bench_integrals(c: &mut Criterion) -> Result<(), Box<dyn Error>> {
    let hydrogen_sto3g = test!("hydrogen", "STO-3G");
    let hydrogen_631g = test!("hydrogen", "6-31G");

    let water_sto3g = test!("water", "STO-3G");
    let water_631g = test!("water", "6-31G");

    let benzene_sto3g = test!("benzene", "STO-3G");
    // let benzene_631g = test!("benzene", "6-31G");

    // bench_overlap(
    //     c,
    //     &[
    //         &hydrogen_sto3g,
    //         &hydrogen_631g,
    //         &water_sto3g,
    //         &water_631g,
    //         &benzene_sto3g,
    //         &benzene_631g,
    //     ],
    // )
    // .unwrap();

    // bench_kinetic(
    //     c,
    //     &[
    //         &hydrogen_sto3g,
    //         &hydrogen_631g,
    //         &water_sto3g,
    //         &water_631g,
    //         &benzene_sto3g,
    //         &benzene_631g,
    //     ],
    // )
    // .unwrap();

    // TODO: once it's faster, add benzene to electron benchese
    bench_electron(
        c,
        &[&hydrogen_sto3g, &hydrogen_631g, &water_sto3g, &water_631g],
    )
    .unwrap();

    Ok(())
}

criterion_group!(benches, bench_integrals);
criterion_main!(benches);
