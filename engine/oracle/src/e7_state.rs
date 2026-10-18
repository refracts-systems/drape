use crate::io::{write_text, Json};
use crate::jsonscan::{number, string};
use crate::linalg::Vec3;
use std::fs;
use std::path::{Path, PathBuf};

/// One equilibrium (or start) state of a sheet: positions of every vertex and the step size the solve ended with.
/// Stored as <prefix>.json {"vertices", "final_h", "dtype"} + <prefix>.bin (little endian f32 or f64, x y z per vertex).
pub struct State {
    pub positions: Vec<Vec3>,
    pub final_h: f64,
    pub single: bool,
}

fn with_extension(prefix: &Path, extension: &str) -> PathBuf {
    let mut name = prefix.as_os_str().to_owned();
    name.push(".");
    name.push(extension);
    PathBuf::from(name)
}

pub fn write_state(prefix: &Path, state: &State) -> Result<(), String> {
    let bytes: Vec<u8> = state.positions.iter().flatten().flat_map(|&v| if state.single { (v as f32).to_le_bytes().to_vec() } else { v.to_le_bytes().to_vec() }).collect();
    fs::write(with_extension(prefix, "bin"), bytes).map_err(|e| e.to_string())?;
    let json = Json::new().integer("vertices", state.positions.len()).number("final_h", state.final_h).text("dtype", if state.single { "f32" } else { "f64" });
    write_text(&with_extension(prefix, "json"), &json.render())
}

pub fn read_state(prefix: &Path) -> Result<State, String> {
    let text = fs::read_to_string(with_extension(prefix, "json")).map_err(|e| format!("{}: {e}", prefix.display()))?;
    let bytes = fs::read(with_extension(prefix, "bin")).map_err(|e| format!("{}: {e}", prefix.display()))?;
    let single = string(&text, "", "dtype") == "f32";
    let values: Vec<f64> = if single {
        bytes.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap()) as f64).collect()
    } else {
        bytes.chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().unwrap())).collect()
    };
    let positions = values.chunks_exact(3).map(|p| [p[0], p[1], p[2]]).collect();
    Ok(State { positions, final_h: number(&text, "", "final_h"), single })
}
