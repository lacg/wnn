//! Flow name -> the worker's checkpoint folder name (IDS-81).
//!
//! Twin of the worker's `src/wnn/ram/experiments/flow_dir_name.py`
//! (lower-case; ' ' and '/' become '_'; every other character kept). Both are
//! pinned by `src/wnn/ram/experiments/flow_dir_name_cases.json`. The old
//! dashboard rule mapped every non-[a-z0-9_-] character to '_', so a name with
//! '.' (`...rate0.25...`) never matched the worker's folder and a "restart from
//! beginning" left the stale checkpoints in place.
//!
//! SECURITY: the result is joined under the checkpoints root and deleted
//! recursively. With '/' mapped away it is always ONE path component; the only
//! components that escape or alias the root are "", "." and "..", so those
//! (and a NUL) are refused instead of renamed.

/// The worker's folder name for `flow_name`, or None when it is not a single
/// safe path component (the caller then skips the deletion).
pub fn worker_checkpoint_dir_name(flow_name: &str) -> Option<String>
{
	let name = flow_name.to_lowercase().replace([' ', '/'], "_");
	match name.as_str()
	{
		"" | "." | ".." => None,
		n if n.contains('\0') => None,
		_ => Some(name),
	}
}

#[cfg(test)]
mod tests
{
	use super::*;

	#[test]
	fn matches_the_worker_rule_on_the_shared_cases()
	{
		let raw = include_str!("../../../src/wnn/ram/experiments/flow_dir_name_cases.json");
		let doc: serde_json::Value = serde_json::from_str(raw).expect("cases json");
		let cases = doc["cases"].as_array().expect("cases array");
		assert!(!cases.is_empty());
		for case in cases
		{
			let (name, want) = (case[0].as_str().unwrap(), case[1].as_str().map(str::to_string));
			assert_eq!(worker_checkpoint_dir_name(name), want, "flow name {name:?}");
		}
	}

	#[test]
	fn dotted_rate_names_keep_their_dot()
	{
		let got = worker_checkpoint_dir_name("IDS6-ciciot46m-B15AC-desir-r20419-rate0.25-abi30");
		assert_eq!(got.as_deref(), Some("ids6-ciciot46m-b15ac-desir-r20419-rate0.25-abi30"));
	}
}
