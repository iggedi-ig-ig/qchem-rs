//! McMurchie Davidon integration scheme.
//! Reference:
//!
//! [1] Goings, J. Integrals. https://joshuagoings.com/2017/04/28/integrals/
use nalgebra::Vector3;

use crate::{
    atom::Atom,
    basis::{BasisFunction, BasisFunctionType, ContractedGaussian, Gaussian},
};

use super::{
    utils::{coulomb_auxiliary, hermite_expansion},
    Integrator,
};

#[derive(Default)]
pub struct McMurchieDavidson;

impl Integrator for McMurchieDavidson {
    type Function = BasisFunction;

    fn overlap(&self, functions: (&Self::Function, &Self::Function)) -> f64 {
        let (basis_a, basis_b) = functions;
        let diff = basis_b.position - basis_a.position;

        match (&basis_a.function_type, &basis_b.function_type) {
            (
                BasisFunctionType::ContractedGaussian(ContractedGaussian(data_a)),
                BasisFunctionType::ContractedGaussian(ContractedGaussian(data_b)),
            ) => {
                let mut output = 0.0;
                for (&primitive_a, &primitive_b) in itertools::iproduct!(data_a, data_b) {
                    output += primitive_a.coefficient
                        * primitive_b.coefficient
                        * primitive_overlap(primitive_a, primitive_b, diff);
                }
                output
            }
        }
    }

    fn kinetic(&self, functions: (&Self::Function, &Self::Function)) -> f64 {
        let (basis_a, basis_b) = functions;
        let diff = basis_b.position - basis_a.position;

        match (&basis_a.function_type, &basis_b.function_type) {
            (
                BasisFunctionType::ContractedGaussian(ContractedGaussian(data_a)),
                BasisFunctionType::ContractedGaussian(ContractedGaussian(data_b)),
            ) => {
                let mut output = 0.0;
                for (&primitive_a, &primitive_b) in itertools::iproduct!(data_a, data_b) {
                    output += primitive_a.coefficient
                        * primitive_b.coefficient
                        * primitive_kinetic(primitive_a, primitive_b, diff);
                }
                output
            }
        }
    }

    fn nuclear(&self, functions: (&Self::Function, &Self::Function), nuclei: &[Atom]) -> f64 {
        let (basis_a, basis_b) = functions;
        let diff = basis_b.position - basis_a.position;

        match (&basis_a.function_type, &basis_b.function_type) {
            (
                BasisFunctionType::ContractedGaussian(ContractedGaussian(data_a)),
                BasisFunctionType::ContractedGaussian(ContractedGaussian(data_b)),
            ) => {
                let mut output = 0.0;
                // TODO: experiment with order of iteration - maybe it's faster to iterate through nuclei in the inner-most loop
                for (nucleus, &primitive_a, &primitive_b) in
                    itertools::iproduct!(nuclei, data_a, data_b)
                {
                    let product_center = product_center(
                        basis_a.position,
                        primitive_a.exponent,
                        basis_b.position,
                        primitive_b.exponent,
                    );

                    output += primitive_a.coefficient
                        * primitive_b.coefficient
                        * primitive_nuclear(primitive_a, primitive_b, diff, product_center, nucleus)
                }

                output
            }
        }
    }

    fn electron_repulsion(
        &self,
        functions: (
            &Self::Function,
            &Self::Function,
            &Self::Function,
            &Self::Function,
        ),
    ) -> f64 {
        let (basis_a, basis_b, basis_c, basis_d) = functions;
        let diff_ab = basis_b.position - basis_a.position;
        let diff_cd = basis_d.position - basis_c.position;

        match (
            &basis_a.function_type,
            &basis_b.function_type,
            &basis_c.function_type,
            &basis_d.function_type,
        ) {
            (
                BasisFunctionType::ContractedGaussian(ContractedGaussian(data_a)),
                BasisFunctionType::ContractedGaussian(ContractedGaussian(data_b)),
                BasisFunctionType::ContractedGaussian(ContractedGaussian(data_c)),
                BasisFunctionType::ContractedGaussian(ContractedGaussian(data_d)),
            ) => {
                let mut output = 0.0;
                for &primitive_a in data_a {
                    for &primitive_b in data_b {
                        for &primitive_c in data_c {
                            for &primitive_d in data_d {
                                let product_center_ab = product_center(
                                    basis_a.position,
                                    primitive_a.exponent,
                                    basis_b.position,
                                    primitive_b.exponent,
                                );

                                let product_center_cd = product_center(
                                    basis_c.position,
                                    primitive_c.exponent,
                                    basis_d.position,
                                    primitive_d.exponent,
                                );

                                let diff_product = product_center_cd - product_center_ab;

                                output += primitive_a.coefficient
                                    * primitive_b.coefficient
                                    * primitive_c.coefficient
                                    * primitive_d.coefficient
                                    * primitive_electron(
                                        primitive_a,
                                        primitive_b,
                                        primitive_c,
                                        primitive_d,
                                        diff_ab,
                                        diff_cd,
                                        diff_product,
                                    )
                            }
                        }
                    }
                }

                output
            }
        }
    }
}

fn primitive_overlap(primitive_a: Gaussian, primitive_b: Gaussian, diff: Vector3<f64>) -> f64 {
    let Gaussian {
        exponent: exp_a,
        angular: (l1, m1, n1),
        ..
    } = primitive_a;

    let Gaussian {
        exponent: exp_b,
        angular: (l2, m2, n2),
        ..
    } = primitive_b;

    hermite_expansion([l1, l2, 0], diff.x, exp_a, exp_b)
        * hermite_expansion([m1, m2, 0], diff.y, exp_a, exp_b)
        * hermite_expansion([n1, n2, 0], diff.z, exp_a, exp_b)
        * (std::f64::consts::PI / (exp_a + exp_b)).powi(3).sqrt()
}

fn primitive_kinetic(primitive_a: Gaussian, primitive_b: Gaussian, diff: Vector3<f64>) -> f64 {
    let Gaussian {
        exponent: b_exp,
        angular: (l, m, n),
        ..
    } = primitive_b;

    let angular_step =
        |i, j, k| primitive_overlap(primitive_a, add_angular(primitive_b, [i, j, k]), diff);

    let term_0 =
        b_exp * (2 * (l + m + n) + 3) as f64 * primitive_overlap(primitive_a, primitive_b, diff);
    let term_1 = -2.0
        * b_exp.powi(2)
        * (angular_step(2, 0, 0) + angular_step(0, 2, 0) + angular_step(0, 0, 2));
    let term_2 = -0.5
        * ((l * (l - 1)) as f64 * angular_step(-2, 0, 0)
            + (m * (m - 1)) as f64 * angular_step(0, -2, 0)
            + (n * (n - 1)) as f64 * angular_step(0, 0, -2));
    term_0 + term_1 + term_2
}

fn primitive_nuclear(
    primitive_a: Gaussian,
    primitive_b: Gaussian,
    // difference of the positions of the two basis functions: b - a
    diff: Vector3<f64>,
    // the product center of the two basis functions
    product_center: Vector3<f64>,
    nucleus: &Atom,
) -> f64 {
    let Gaussian {
        exponent: a,
        angular: (l1, m1, n1),
        ..
    } = primitive_a;

    let Gaussian {
        exponent: b,
        angular: (l2, m2, n2),
        ..
    } = primitive_b;

    let p = a + b;
    let diff_nucleus = nucleus.position - product_center;

    let mut sum = 0.0;
    for (t, u, v) in itertools::iproduct!(0..=l1 + l2, 0..=m1 + m2, 0..=n1 + n2) {
        // TODO: if this was a nested loop, e1 and e2 would not have to be calculated in the inner most branch
        //  which could potentially speed up computation significantly. It depends on what the compiler does.

        let e1 = hermite_expansion([l1, l2, t], diff.x, a, b);
        let e2 = hermite_expansion([m1, m2, u], diff.y, a, b);
        let e3 = hermite_expansion([n1, n2, v], diff.z, a, b);
        sum += e1 * e2 * e3 * coulomb_auxiliary(t, u, v, 0, p, diff_nucleus)
    }
    (-nucleus.charge() as f64 * std::f64::consts::TAU / p) * sum
}

fn primitive_electron(
    primitive_a: Gaussian,
    primitive_b: Gaussian,
    primitive_c: Gaussian,
    primitive_d: Gaussian,
    diff_ab: Vector3<f64>,
    diff_cd: Vector3<f64>,
    diff_product: Vector3<f64>,
) -> f64 {
    let Gaussian {
        exponent: a,
        angular: (l1, m1, n1),
        ..
    } = primitive_a;
    let Gaussian {
        exponent: b,
        angular: (l2, m2, n2),
        ..
    } = primitive_b;
    let Gaussian {
        exponent: c,
        angular: (l3, m3, n3),
        ..
    } = primitive_c;
    let Gaussian {
        exponent: d,
        angular: (l4, m4, n4),
        ..
    } = primitive_d;

    let p = a + b;
    let q = c + d;
    let alpha = p * q / (p + q);

    let mut sum = 0.0;
    for t1 in 0..=l1 + l2 {
        for u1 in 0..=m1 + m2 {
            for v1 in 0..=n1 + n2 {
                for t2 in 0..=l3 + l4 {
                    for u2 in 0..=m3 + m4 {
                        for v2 in 0..=n3 + n4 {
                            let e1 = hermite_expansion([l1, l2, t1], diff_ab.x, a, b);
                            let e2 = hermite_expansion([m1, m2, u1], diff_ab.y, a, b);
                            let e3 = hermite_expansion([n1, n2, v1], diff_ab.z, a, b);
                            let e4 = hermite_expansion([l3, l4, t2], diff_cd.x, c, d);
                            let e5 = hermite_expansion([m3, m4, u2], diff_cd.y, c, d);
                            let e6 = hermite_expansion([n3, n4, v2], diff_cd.z, c, d);

                            sum += e1
                                * e2
                                * e3
                                * e4
                                * e5
                                * e6
                                * coulomb_auxiliary(
                                    t1 + t2,
                                    u1 + u2,
                                    v1 + v2,
                                    0,
                                    alpha,
                                    diff_product,
                                )
                                * if (t2 + u2 + v2) % 2 == 0 { 1.0 } else { -1.0 }
                            // (-1)^(t2 + u2 + v2)
                        }
                    }
                }
            }
        }
    }

    2.0 * std::f64::consts::PI.powi(5).sqrt() * (p * q * (p + q).sqrt()).recip() * sum
}

#[inline(always)]
fn add_angular(gaussian: Gaussian, [i, j, k]: [i32; 3]) -> Gaussian {
    let Gaussian {
        angular: (l, m, n), ..
    } = gaussian;

    Gaussian {
        angular: (l + i, m + j, n + k),
        ..gaussian
    }
}

#[inline(always)]
fn product_center(
    a_pos: Vector3<f64>,
    a_exp: f64,
    b_pos: Vector3<f64>,
    b_exp: f64,
) -> Vector3<f64> {
    (a_exp * a_pos + b_exp * b_pos) / (a_exp + b_exp)
}

#[cfg(tests)]
mod tests {
    use nalgebra::Vector3;

    use crate::basis::Gaussian;

    #[test]
    fn test_primitive_overlap() {
        let trivial = Gaussian {
            exponent: 1.0,
            coefficient: 1.0,
            angular: (0, 0, 0),
        };
        let angular_i = Gaussian {
            exponent: 1.0,
            coefficient: 1.0,
            angular: (1, 0, 0),
        };
        let angular_j = Gaussian {
            exponent: 1.0,
            coefficient: 1.0,
            angular: (0, 1, 0),
        };
        let angular_k = Gaussian {
            exponent: 1.0,
            coefficient: 1.0,
            angular: (0, 0, 1),
        };

        assert_eq!(
            super::primitive_overlap(trivial, trivial, Vector3::new(1.0, 0.0, 0.)),
            1.194077663824459,
            "trivial overlap (all parameters one) is incorrect"
        );
        assert_eq!(
            super::primitive_overlap(angular_i, angular_i, Vector3::new(1.0, 0.0, 0.)),
            0.0,
            "overlap for two trivial p_x orbitals is wrong"
        );
        assert_eq!(
            super::primitive_overlap(angular_j, angular_j, Vector3::new(1.0, 0.0, 0.)),
            0.29851941595611475,
            "overlap for two trivial p_y orbitals is wrong"
        );
        assert_eq!(
            super::primitive_overlap(angular_k, angular_k, Vector3::new(1.0, 0.0, 0.)),
            0.29851941595611475,
            "overlap for two trivial p_z orbitals is wrong"
        );
    }
}
