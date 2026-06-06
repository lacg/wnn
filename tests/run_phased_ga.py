#!/usr/bin/env python
"""Thin CLI wrapper for the phased-GA controller search.

The implementation now lives in `wnn.control.phased_ga` (moved out of tests/ so
production code — the dashboard worker's flow_runner — can import the callables
without a sys.path hack). This wrapper preserves the historical
`python tests/run_phased_ga.py ...` invocation used by the detached controller
runs and the resume recipes; it is behaviourally identical to the old script.

All flags, stages, signal/emergency handling, and outputs are unchanged — see
`wnn.control.phased_ga` for the full module docstring and CLI.
"""
from wnn.control.phased_ga import main

if __name__ == "__main__":
    main()
