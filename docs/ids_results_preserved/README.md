# Preserved hand-written blocks for `docs/ids_results.md`

**These files are the DURABLE SOURCE for the hand-written sections of the canonical IDS
results doc.** The doc itself is hand-assembled from several generators' stdout; nothing
in `scripts/` produces it end-to-end (verified 24/08/2026 — no script emits its
"canonical source of truth" header). So every regeneration risks dropping the sections
below, which is why they live here rather than only inline.

**POSITION IS THE FILENAME PREFIX = section number x 10:**

```
00-09   HEAD  -- emitted BEFORE the generated sections
10-59         -- reserved for GENERATED sections 1-5; no files live here
60-99   TAIL  -- emitted AFTER the generated sections
```
A file landing in 10-59 is a misfile; `orphaned_blocks()` reports it loudly rather than
dropping it silently.

| file | section | position | why it is hand-written |
|---|---|---|---|
| `00_paper_baseline_comparison.md` | 0 | head | the paper claim + measured RF/XGB baselines, hand-assembled from the rollup |
| `60_config_lock_analysis_09aug.md` | 6 | tail | analysis prose written 09/08/2026 |
| `70_46m_single_flow_manual.md` | 7 | tail | single 46M flow, not covered by `build_xds_5tables.py` |
| `80_idsx_acce_interim_n2.md` | 8 | tail | INTERIM n=2; carries three CORRECTIONS to claims in sections 0-7 |

Loader: `scripts/ids_results_preserved.py` — `head_blocks()`, `tail_blocks()`,
`orphaned_blocks()`, `guard_canonical_target()`.

**Only the end-to-end assembler may compose these.** A generator that emits a DIFFERENT
report must not append them — that injects one document's sections into another. That was a
real bug introduced and caught here on 24/08/2026.

## To regenerate the doc safely

There is no one-shot assembler. Rebuild the generated sections per the provenance table at
the top of `docs/ids_results.md`, then append these files in order:

```bash
# head blocks, then the generated sections, then tail blocks
for f in docs/ids_results_preserved/0[0-9]_*.md; do cat "$f"; printf '\n\n'; done  >  /tmp/doc.md
#   ... generated sections 1-5 here ...                                             >> /tmp/doc.md
for f in docs/ids_results_preserved/[6-9][0-9]_*.md; do printf '\n\n'; cat "$f"; done >> /tmp/doc.md
```

## ⚠️ Do NOT run `build_oi_vs_old_report.py --out docs/ids_results.md`

It does **not** produce this document — it emits a different report entirely
(`# <prefix> — OI-v2 vs OLD baseline`) and `write_text()` would replace all ~11,400 lines,
including sections 0-5, not merely these blocks. The script now refuses that path unless
`--force-overwrite-canonical` is passed. Section 7's own header has warned about this since
it was written ("a full regen of this file may drop this section — re-append from git if so").
