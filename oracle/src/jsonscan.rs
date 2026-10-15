/// Flat key lookup for the small JSON files this program and the GPU runner write.
/// Finds the first `"key":` after the first occurrence of `"anchor"` (or from the start when anchor is empty).
fn raw<'a>(text: &'a str, anchor: &str, key: &str) -> Option<&'a str> {
    let start = if anchor.is_empty() { 0 } else { text.find(&format!("\"{anchor}\""))? };
    let marker = format!("\"{key}\":");
    let at = start + text[start..].find(&marker)? + marker.len();
    let rest = text[at..].trim_start();
    let end = match rest.strip_prefix('"') {
        Some(quoted) => quoted.find('"')? + 2,
        None => rest.find([',', '}', '\n']).unwrap_or(rest.len()),
    };
    Some(rest[..end].trim())
}

/// Missing or `null` values come back as NaN, which the Json writer renders as null.
pub fn number(text: &str, anchor: &str, key: &str) -> f64 {
    raw(text, anchor, key).and_then(|v| v.parse().ok()).unwrap_or(f64::NAN)
}

pub fn string(text: &str, anchor: &str, key: &str) -> String {
    raw(text, anchor, key).map(|v| v.trim_matches('"').to_string()).unwrap_or_default()
}

pub fn flag(text: &str, anchor: &str, key: &str) -> bool {
    raw(text, anchor, key) == Some("true")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_nested_and_spaced_values() {
        let text = "{\"a\": {\"iterations\": 7}, \"b\": {\"iterations\": 9, \"gpu_ms\": null, \"name\": \"x, y\"}, \"ok\": true}";
        assert_eq!(number(text, "b", "iterations"), 9.0);
        assert_eq!(number(text, "", "iterations"), 7.0);
        assert!(number(text, "", "gpu_ms").is_nan());
        assert_eq!(string(text, "", "name"), "x, y");
        assert!(flag(text, "", "ok"));
    }
}
