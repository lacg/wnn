"""Flow name -> on-disk folder name (checkpoints/<name>, logs/<name>_HHMMSS.log).

THE rule the worker names a flow's checkpoint folder by. The dashboard's
"restart from beginning" deletes that folder, so its Rust twin
(dashboard/src/api/checkpoint_dir_name.rs) must produce the SAME name; both are
pinned against flow_dir_name_cases.json (IDS-81: the dashboard mapped '.' to
'_', so `...rate0.25...` never matched the worker's folder and the stale
checkpoints survived a from-zero restart).
"""


def flow_dir_name(flow_name: str) -> str:
	"""Lower-case; ' ' and '/' become '_'; every other character is kept."""
	return flow_name.lower().replace(" ", "_").replace("/", "_")
