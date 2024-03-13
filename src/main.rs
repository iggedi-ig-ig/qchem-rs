use atom::Atom;
use clap::{ArgAction, Parser, Subcommand};

mod atom;
mod basis;
mod integrals;
mod molecule;
mod utils;

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
    HartreeFock {/* specify what to output */},
}

fn main() {
    let args: Args = Args::parse();

    println!("{args:?}");
}
