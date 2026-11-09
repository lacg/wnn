"""FlowWorker takes its tracker DB EXPLICITLY (08/11/2026).

The old default (db/wnn.db under the CWD) meant any script reusing FlowWorker
as a revalidation host silently opened the LIVE DB when run from the repo root.
Only the production entry points (worker main, flow_runner) pass live_db_path().
"""

import pytest

from wnn.ram.experiments.worker import FlowWorker


def test_no_db_path_is_refused():
	with pytest.raises(TypeError):
		FlowWorker()
	with pytest.raises(ValueError):
		FlowWorker(db_path="")


def test_scripts_can_run_without_the_live_db():
	assert FlowWorker(db_path=":memory:").tracker is not None


def test_production_entry_points_pass_the_live_db():
	import inspect
	from wnn.ram.experiments import flow_runner, worker
	assert "db_path=live_db_path()" in inspect.getsource(flow_runner.run_one_flow)
	assert "db_path=live_db_path()" in inspect.getsource(worker.main)
