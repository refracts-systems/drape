use std::fs;
use std::path::Path;

/// Little-endian arrays concatenated with 8-byte alignment, described by a JSON manifest.
pub struct BinaryArrays {
    bytes: Vec<u8>,
    entries: Vec<String>,
}

impl BinaryArrays {
    pub fn new() -> BinaryArrays {
        BinaryArrays { bytes: Vec::new(), entries: Vec::new() }
    }

    pub fn push_u32(&mut self, name: &str, values: impl Iterator<Item = u32>) {
        let offset = self.align();
        let before = self.bytes.len();
        values.for_each(|v| self.bytes.extend_from_slice(&v.to_le_bytes()));
        self.describe(name, "u32", (self.bytes.len() - before) / 4, offset);
    }

    pub fn push_f32(&mut self, name: &str, values: impl Iterator<Item = f32>) {
        let offset = self.align();
        let before = self.bytes.len();
        values.for_each(|v| self.bytes.extend_from_slice(&v.to_le_bytes()));
        self.describe(name, "f32", (self.bytes.len() - before) / 4, offset);
    }

    pub fn push_f64(&mut self, name: &str, values: &[f64]) {
        let offset = self.align();
        values.iter().for_each(|v| self.bytes.extend_from_slice(&v.to_le_bytes()));
        self.describe(name, "f64", values.len(), offset);
    }

    fn align(&mut self) -> usize {
        while self.bytes.len() % 8 != 0 {
            self.bytes.push(0);
        }
        self.bytes.len()
    }

    fn describe(&mut self, name: &str, dtype: &str, length: usize, offset: usize) {
        self.entries.push(format!(
            "{{\"name\":\"{name}\",\"dtype\":\"{dtype}\",\"length\":{length},\"offset\":{offset}}}"
        ));
    }

    pub fn manifest(&self) -> String {
        format!("[{}]", self.entries.join(","))
    }

    pub fn write(&self, path: &Path) -> Result<(), String> {
        fs::write(path, &self.bytes).map_err(|e| format!("{}: {e}", path.display()))
    }
}

/// Reads one array described by a manifest written by BinaryArrays.
pub fn read_f64(bytes: &[u8], offset: usize, length: usize) -> Vec<f64> {
    bytes[offset..offset + 8 * length]
        .chunks_exact(8)
        .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

pub fn read_u32(bytes: &[u8], offset: usize, length: usize) -> Vec<u32> {
    bytes[offset..offset + 4 * length]
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

pub fn read_f32_file(path: &Path) -> Result<Vec<f32>, String> {
    let bytes = fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(bytes.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect())
}

/// Minimal JSON object writer: values are inserted pre-rendered.
pub struct Json {
    fields: Vec<String>,
}

impl Json {
    pub fn new() -> Json {
        Json { fields: Vec::new() }
    }

    pub fn number(mut self, key: &str, value: f64) -> Json {
        self.fields.push(format!("\"{key}\":{}", render_number(value)));
        self
    }

    pub fn integer(mut self, key: &str, value: usize) -> Json {
        self.fields.push(format!("\"{key}\":{value}"));
        self
    }

    pub fn text(mut self, key: &str, value: &str) -> Json {
        self.fields.push(format!("\"{key}\":\"{}\"", value.replace('"', "\\\"")));
        self
    }

    pub fn flag(mut self, key: &str, value: bool) -> Json {
        self.fields.push(format!("\"{key}\":{value}"));
        self
    }

    pub fn raw(mut self, key: &str, rendered: String) -> Json {
        self.fields.push(format!("\"{key}\":{rendered}"));
        self
    }

    /// The fields of `self` followed by those of `other`.
    pub fn extend(mut self, other: Json) -> Json {
        self.fields.extend(other.fields);
        self
    }

    pub fn render(&self) -> String {
        format!("{{{}}}", self.fields.join(","))
    }
}

pub fn render_number(value: f64) -> String {
    if value.is_finite() { format!("{value:e}") } else { "null".to_string() }
}

pub fn render_history(history: &[(usize, f64)]) -> String {
    let items: Vec<String> = history.iter().map(|(i, r)| format!("[{i},{}]", render_number(*r))).collect();
    format!("[{}]", items.join(","))
}

pub fn write_text(path: &Path, text: &str) -> Result<(), String> {
    fs::write(path, text).map_err(|e| format!("{}: {e}", path.display()))
}

/// Pulls `"offset":N` and `"length":N` for a named array out of a manifest string.
pub fn locate(manifest: &str, name: &str) -> Result<(usize, usize), String> {
    let key = format!("\"name\":\"{name}\"");
    let start = manifest.find(&key).ok_or(format!("array '{name}' missing from manifest"))?;
    let entry = &manifest[start..manifest[start..].find('}').map(|e| start + e).unwrap_or(manifest.len())];
    Ok((field(entry, "offset")?, field(entry, "length")?))
}

fn field(entry: &str, key: &str) -> Result<usize, String> {
    let marker = format!("\"{key}\":");
    let start = entry.find(&marker).ok_or(format!("{key} missing"))? + marker.len();
    let digits: String = entry[start..].chars().take_while(|c| c.is_ascii_digit()).collect();
    digits.parse().map_err(|_| format!("bad {key}"))
}
