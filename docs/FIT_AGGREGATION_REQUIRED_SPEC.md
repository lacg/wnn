# Make `--fit-aggregation` REQUIRED (post-sweep change)

**Status:** proposed 24/08/2026, awaiting Luiz's approval on the final form.
**Precondition: NO controller chain may be running when this lands.** `phased_ga.py` is
imported fresh by each run's subprocess, so changing argparse mid-chain silently splits a
cohort — runs before and after the edit fly under different fitness resolution. The gated
weight sweep (PID 35700) must finish first.

---

## The problem

`src/wnn/ram/../wnn/control/phased_ga.py:2783`

```python
ap.add_argument("--fit-aggregation", choices=["harmonic", "arithmetic", "zscore"],
                default=None,
                help="... Unset = harmonic in-search + arithmetic stage-select "
                     "(the legacy/banked behavior).")
```

**An absent `--fit-aggregation` is not a no-op — it is a behaviour.** Its unset state resolves
to the legacy combine. A script that simply never mentions the flag reads as "fine, defaults are
sensible" and silently measures under the compass we replaced on 19/08.

This nearly cost a cohort. `scripts/sweep_ladder_chain.sh` (46-52 runs, ~150-170 h) carried no
`--fit-aggregation` at all. It would have produced a complete, plausible, publishable bits curve
measured under the legacy combine while the sweep selecting its weights flew `zscore`, and
**nothing in its output would have said so.** Caught 24/08 only because the weights were being
refreshed by hand.

### This is a recurring failure family, not a one-off

| knob | unset/ignored state | how it was found |
|---|---|---|
| `--fit-weight-alt` | silently a NO-OP at 4 of 5 sites | 2 sweep arms void; arm bit-identical to its control |
| `ga_generations` | dead at RUN time (`experiments.max_iterations` always populated) | found before it cost anything |
| `--fit-aggregation` | resolves to the legacy combine | caught by hand, 24/08 |

**A knob whose unset state is a behaviour rather than an error will eventually cost a cohort.**

---

## Blast radius (measured 24/08/2026)

```
scripts invoking wnn.control.phased_ga           121
  of which DECLARE --fit-aggregation               3   (gated_weight_sweep_chain,
                                                        fitness_agg_ab_chain, sweep_ladder_chain)
  of which are silent                            118
callers modified in the last 14 days              47
```

So a naive `required=True` breaks 118 scripts at once — including archival ones we may
legitimately want to re-run to reproduce banked numbers.

---

## The design: required, PLUS an explicit `legacy` value

Making the flag required is only affordable if there is a name for what the archive already
does. Otherwise "required" forces a *semantic* choice onto 118 historical scripts whose correct
answer is "whatever it did when it produced the banked number".

```python
ap.add_argument("--fit-aggregation", required=True,
                choices=["legacy", "harmonic", "arithmetic", "zscore"],
                help="REQUIRED. legacy = harmonic in-search + arithmetic stage-select, "
                     "bit-identical to the pre-24/08 unset default — use it to reproduce "
                     "banked numbers. zscore = winsorized robust z (current regime).")
```

- `legacy` reproduces today's `None` path **exactly**, so no archival script changes meaning.
- No default at all, so a silent script fails LOUDLY at argparse instead of quietly picking a compass.
- Every script then *declares* its compass, and the declaration is greppable.

### Why not just flip the default to `zscore`

It fixes today's footgun by planting a reproducibility one: every archived script that does not
set the flag would silently change behaviour, and re-running `fitness_agg_ab_chain.sh` next month
would no longer reproduce its own banked numbers. Silent drift in the archive is worse than a
loud break.

---

## Migration

1. Add `legacy` as a choice; keep `default=None` mapping to it. **Verify bit-identity** —
   one short run with `--fit-aggregation legacy` vs one with the flag absent must produce
   identical fitness values and identical stage-select output. This is the gate on the whole change.
2. Mechanical pass: append `--fit-aggregation legacy` to every silent caller. It preserves each
   script's banked semantics exactly and is a pure no-op at this step.
3. Flip to `required=True`, remove the default. Now nothing can be silent.
4. Live/forward scripts move to `zscore` deliberately, one at a time, each with its own reason.

Steps 1-2 are safe to land any time the box is idle. Steps 3-4 are the interface change.

---

## Also considered: the viability gate

`--gate-stable` / `--gate-err` are absent from the same 118 scripts. The case is **weaker** —
gate-off is bit-identical, so an absent gate does not silently change a number, it only declines
a filter. Recommend leaving the gate optional and revisiting only if it becomes the default
regime rather than an opt-in.

---

## Verification checklist

- [ ] `legacy` proven bit-identical to unset (fitness values + stage-select choice)
- [ ] all 121 callers declare a value; `grep -L -- '--fit-aggregation'` over callers returns empty
- [ ] no chain running at the moment of the `required=True` flip
- [ ] one smoke run per declared value before any cohort is armed

Relates to: `project_alt_weight_plumbing_bug`, `project_idsz_budget_confound` (the dead
`ga_generations` knob), `feedback_discuss_before_config_change`.
