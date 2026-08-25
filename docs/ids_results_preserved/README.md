# Preserved hand-written blocks for `docs/ids_results.md`

**These files are the DURABLE SOURCE for the hand-written sections of the canonical IDS
results doc.** The doc itself is hand-assembled from several generators' stdout; nothing
in `scripts/` produces it end-to-end (verified 24/08/2026 — no script emits its
"canonical source of truth" header). So every regeneration risks dropping the sections
below, which is why they live here rather than only inline.

Files are appended to the generated content in FILENAME ORDER. The numeric prefix is the
section number x10, leaving room to insert.

| file | section | why it is hand-written |
|---|---|---|
| `60_config_lock_analysis_09aug.md` | 6 | analysis prose written 09/08/2026 |
| `70_46m_single_flow_manual.md` | 7 | single 46M flow, not covered by `build_xds_5tables.py` |
| `80_idsx_acce_interim_n2.md` | 8 | INTERIM n=2; carries three CORRECTIONS to claims in sections 0-7 |

## To regenerate the doc safely

There is no one-shot assembler. Rebuild the generated sections per the provenance table at
the top of `docs/ids_results.md`, then append these files in order:

```bash
for f in docs/ids_results_preserved/[0-9]*.md; do printf '\n\n'; cat "$f"; done >> docs/ids_results.md
```

## ⚠️ Do NOT run `build_oi_vs_old_report.py --out docs/ids_results.md`

It does **not** produce this document — it emits a different report entirely
(`# <prefix> — OI-v2 vs OLD baseline`) and `write_text()` would replace all ~11,400 lines,
including sections 0-5, not merely these blocks. The script now refuses that path unless
`--force-overwrite-canonical` is passed. Section 7's own header has warned about this since
it was written ("a full regen of this file may drop this section — re-append from git if so").
