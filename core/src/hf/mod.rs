pub mod rhf;
mod uhf;
pub(super) mod utils;

pub use rhf::{restricted_hartree_fock, RestrictedHartreeFockOutput};
pub use uhf::{unrestricted_hartree_fock, UnrestrictedHartreeFockOutput};

/// The input to a hartree fock calculation
pub struct HartreeFockConfig {
    /// the maximum number of iterations to try
    pub max_iterations: usize,
    /// the smallest number that isn't treated as zero. For example, if the density
    /// matrix rms changes by less than this, the system is considered converged.
    pub epsilon: f64,
}
