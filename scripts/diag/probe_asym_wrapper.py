# Run the failure probe with disturbance_stream wrapped, to log the exact asym the scorer flew.
import runpy, sys
import wnn.control.evaluator as E
_orig = E.disturbance_stream
def _logged(dist, s):
	d, a = _orig(dist, s)
	print(f"[asym] score_seed={s} asym={[round(float(x), 3) for x in a]}", flush=True)
	return d, a
E.disturbance_stream = _logged
sys.argv = ["probe_episode_failures.py"] + sys.argv[1:]
runpy.run_path("scripts/probe_episode_failures.py", run_name="__main__")
