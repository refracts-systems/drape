mod band;
mod bsr;
mod cases;
mod compare;
mod contact;
mod coupon;
mod criteria;
mod drape;
mod dual;
mod e2_checks;
mod e2_compare;
mod e2_generate;
mod e2_summarize;
mod e3_checks;
mod e3_compare;
mod e3_generate;
mod e3_summarize;
mod e4_compare;
mod e4_diagnose;
mod e4_rules;
mod e4_summarize;
mod e5_checks;
mod e5_compare;
mod e5_summarize;
mod generate;
mod hinge;
mod io;
mod jsonscan;
mod lanczos;
mod linalg;
mod material;
mod membrane;
mod mesh;
mod metrics;
mod newton;
mod pcg;
mod rng;
mod summarize;
#[cfg(test)]
mod testing;

use std::collections::HashMap;
use std::path::PathBuf;
use std::process::ExitCode;

const USAGE: &str = "usage:
  oracle generate --case <tension|slack|dynamic> --n <grid columns> --out <dir> [--tol 1e-7] [--max-iter 20000] [--replace-every 0] [--replace-above 0]
  oracle compare --dir <case dir>
  oracle summarize --dir <cases dir> --out <summary.json>
  oracle e2-generate --path <uniaxial|biaxial|patch> --n <grid columns> --out <dir>
  oracle e2-compare --dir <coupon dir>
  oracle e2-summarize --dir <coupons dir> --out <summary.json>
  oracle e3-generate --n <grid columns> --out <dir> [--reference true|false]
  oracle e3-compare --dir <drape dir>
  oracle e3-summarize --dir <drapes dir> --out <summary.json>
  oracle e4-generate --n <grid columns> --out <dir> [--reference true|false] [--final true|false]
  oracle e4-compare --dir <drape dir> --rung <A|B|C|D>
  oracle e4-summarize --dir <drapes dir> --out <summary.json>
  oracle e4-diagnose --dir <drape dir> --step <k> --out <diagnosis.json>
  oracle e5-generate --n <grid columns> --out <dir> [--reference true|false]
  oracle e5-compare --dir <drape5 dir>
  oracle e5-summarize --dir <drape5 dir> --baseline <experiment4-final.json> --out <summary.json>";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match dispatch(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("error: {message}\n{USAGE}");
            ExitCode::FAILURE
        }
    }
}

fn dispatch(args: &[String]) -> Result<(), String> {
    let (command, rest) = args.split_first().ok_or("missing command")?;
    let flags = parse_flags(rest)?;
    match command.as_str() {
        "generate" => generate::run(&generate_options(&flags)?, &PathBuf::from(required(&flags, "out")?)),
        "compare" => compare::run(&PathBuf::from(required(&flags, "dir")?)),
        "summarize" => summarize::run(&PathBuf::from(required(&flags, "dir")?), &PathBuf::from(required(&flags, "out")?)),
        "e2-generate" => e2_generate::run(coupon::LoadPath::parse(required(&flags, "path")?)?, number(&flags, "n", "33")?, &PathBuf::from(required(&flags, "out")?)),
        "e2-compare" => e2_compare::run(&PathBuf::from(required(&flags, "dir")?)),
        "e2-summarize" => e2_summarize::run(&PathBuf::from(required(&flags, "dir")?), &PathBuf::from(required(&flags, "out")?)),
        "e3-generate" => e3_generate::run(number(&flags, "n", "33")?, number(&flags, "reference", "true")?, &PathBuf::from(required(&flags, "out")?)),
        "e3-compare" => e3_compare::run(&PathBuf::from(required(&flags, "dir")?)),
        "e3-summarize" => e3_summarize::run(&PathBuf::from(required(&flags, "dir")?), &PathBuf::from(required(&flags, "out")?)),
        "e4-generate" => e3_generate::generate(number(&flags, "n", "33")?, number(&flags, "reference", "true")?, &PathBuf::from(required(&flags, "out")?), e4_scheme(number(&flags, "final", "false")?)),
        "e4-compare" => e4_compare::run(&PathBuf::from(required(&flags, "dir")?), required(&flags, "rung")?),
        "e4-summarize" => e4_summarize::run(&PathBuf::from(required(&flags, "dir")?), &PathBuf::from(required(&flags, "out")?)),
        "e5-generate" => e3_generate::generate(number(&flags, "n", "33")?, number(&flags, "reference", "true")?, &PathBuf::from(required(&flags, "out")?), &e3_generate::E5_SCHEME),
        "e5-compare" => e5_compare::run(&PathBuf::from(required(&flags, "dir")?)),
        "e5-summarize" => e5_summarize::run(&PathBuf::from(required(&flags, "dir")?), &PathBuf::from(required(&flags, "baseline")?), &PathBuf::from(required(&flags, "out")?)),
        "e4-diagnose" => e4_diagnose::run(&PathBuf::from(required(&flags, "dir")?), number(&flags, "step", "89")?, &PathBuf::from(required(&flags, "out")?)),
        other => Err(format!("unknown command '{other}'")),
    }
}

fn generate_options(flags: &HashMap<String, String>) -> Result<generate::GenerateOptions, String> {
    Ok(generate::GenerateOptions {
        kind: cases::CaseKind::parse(required(flags, "case")?)?,
        columns: number(flags, "n", "33")?,
        tolerance: number(flags, "tol", "1e-7")?,
        max_iterations: number(flags, "max-iter", "20000")?,
        replace_every: number(flags, "replace-every", "0")?,
        replace_above: number(flags, "replace-above", "0")?,
    })
}

fn parse_flags(args: &[String]) -> Result<HashMap<String, String>, String> {
    args.chunks(2)
        .map(|pair| match pair {
            [key, value] if key.starts_with("--") => Ok((key[2..].to_string(), value.clone())),
            _ => Err(format!("expected '--flag value', got {pair:?}")),
        })
        .collect()
}

fn required<'a>(flags: &'a HashMap<String, String>, key: &str) -> Result<&'a str, String> {
    flags.get(key).map(String::as_str).ok_or(format!("missing --{key}"))
}

fn number<T: std::str::FromStr>(flags: &HashMap<String, String>, key: &str, default: &str) -> Result<T, String> {
    let text = flags.get(key).map(String::as_str).unwrap_or(default);
    text.parse().map_err(|_| format!("invalid --{key} '{text}'"))
}

/// `--final true` selects the E4-final scheme (f64-floor convergence, oracle cap 200, f32-resolution update stop).
fn e4_scheme(final_scheme: bool) -> &'static e3_generate::Scheme {
    if final_scheme { &e3_generate::E4_FINAL_SCHEME } else { &e3_generate::E4_SCHEME }
}
