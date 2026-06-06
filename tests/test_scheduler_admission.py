"""Unit tests for the budget-aware, type-balanced flow admission policy."""
from wnn.ram.experiments.scheduler import admit, flow_cores, flow_type, detect_budget


def _flow(fid, arch="ids", threads=None):
    p = {"architecture_type": arch}
    if threads is not None:
        p["wnn_num_threads"] = threads
    return {"id": fid, "config": {"params": p}}


def _ids(fid, threads=10):   return _flow(fid, "ids", threads)
def _ctrl(fid, threads=3):   return _flow(fid, "controller", threads)


def test_core_defaults_and_override():
    assert flow_cores(_ids(1)) == 10
    assert flow_cores(_ctrl(2)) == 3
    assert flow_cores(_flow(3, "ids")) == 10           # default by type
    assert flow_cores(_ids(4, threads=5)) == 5         # explicit override
    assert flow_type(_ctrl(5)) == "controller"


def test_both_types_queued_lands_on_1ids_1ctrl():
    # budget 13: 1 IDS(10) + 1 ctrl(3) = 13, balanced. NOT 2 IDS (20>13).
    q = [_ids(1), _ids(2), _ctrl(3), _ctrl(4)]
    got = admit(q, running=[], budget=13)
    ids = [f for f in got if flow_type(f) == "ids"]
    ctrl = [f for f in got if flow_type(f) == "controller"]
    assert len(ids) == 1 and len(ctrl) == 1, [f["id"] for f in got]
    # FIFO within type → oldest ids picked
    assert ids[0]["id"] == 1 and ctrl[0]["id"] == 3


def test_ids_first_when_underrepresented():
    # a controller is already running; an IDS finished freeing 10 cores.
    # Must admit IDS (under-represented), NOT a 4th controller.
    running = [{"id": 3, "type": "controller", "cores": 3}]
    q = [_ids(1), _ctrl(4), _ctrl(5), _ctrl(6)]
    got = admit(q, running, budget=13)  # remaining = 10
    assert [f["id"] for f in got] == [1], [f["id"] for f in got]
    assert flow_type(got[0]) == "ids"


def test_only_controllers_fill_budget():
    # No IDS queued → controllers fill: 13 // 3 = 4 controllers.
    q = [_ctrl(i) for i in range(1, 7)]
    got = admit(q, running=[], budget=13)
    assert len(got) == 4, [f["id"] for f in got]
    assert [f["id"] for f in got] == [1, 2, 3, 4]       # oldest-first


def test_nothing_fits_returns_empty():
    running = [{"id": 9, "type": "ids", "cores": 10}]
    q = [_ids(1)]                                        # needs 10, only 3 free
    assert admit(q, running, budget=13) == []


def test_skips_already_running():
    running = [{"id": 1, "type": "ids", "cores": 10}]
    q = [_ids(1), _ctrl(2)]                              # id 1 already running
    got = admit(q, running, budget=13)
    assert [f["id"] for f in got] == [2]


def test_balance_holds_across_a_full_cycle():
    # IDS gone, 1 ctrl running, both types queued → IDS admitted to rebalance.
    running = [{"id": 100, "type": "controller", "cores": 3}]
    q = [_ids(50), _ctrl(60)]
    got = admit(q, running, budget=13)                  # remaining 10
    # under-rep type is ids(0) vs controller(1) → ids first; then 0 budget left
    assert [f["id"] for f in got] == [50]


def test_detect_budget_dynamic():
    assert detect_budget(reserve=3, cpu_cores=16) == 13
    assert detect_budget(reserve=3, cpu_cores=8) == 5
    assert detect_budget(reserve=5, cpu_cores=4) == 1   # floored at 1
