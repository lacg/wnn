# Code Style Migration Plan — Allman braces, TAB indentation, no optional semicolons

**Author:** code-style specialist (Andrew Martin)
**Date surveyed:** 17/08/2026
**Status:** SURVEY + PLAN. Zero source files were modified to produce this document.
**Approval required before any unit below is executed.**

---

## 0.0 CORRECTIONS — verified by experiment, 18/08/2026

⚠️ **The survey's two headline recommendations were both "accept K&R". Luiz has ruled
that out explicitly ("a very HARD no ... not on my watch"), and both were also wrong on
the facts.** The toolchains were installed and tested on scratch copies; every claim
below is a run, not a reading. Where this section and the body disagree, THIS SECTION
WINS — §3.1-A, §3.2-A and §7 items 1-2 are superseded.

**Allman is achievable on every surface. No language has to accept K&R.**

| surface | verdict | tool, verified |
|---|---|---|
| **Rust** | ✅ full Allman + tabs | `rustup run nightly rustfmt`, config below |
| **TypeScript / Svelte** | ✅ full Allman + tabs + no semicolons | **ESLint + `@stylistic`** — the survey never considered it |
| **Metal** | ✅ full Allman + tabs | `clang-format --assume-filename=x.cpp`, exit 0 on a real shader |
| **Python** | ✅ n/a (no braces) | `ruff format` with `indent-style = "tab"` |

**1. Rust needs TWO options, not one.** `brace_style` governs only ITEMS (fn/struct/impl).
Control flow is governed separately by `control_brace_style`. Setting only the first
gives *partial* Allman — the fn brace moves, `if`/`else` do not — which is worse than
not starting. Verified working config:

```toml
hard_tabs           = true
brace_style         = "AlwaysNextLine"
control_brace_style = "AlwaysNextLine"
match_arm_blocks    = true
tab_spaces          = 2   # see "tab WIDTH" below — rustfmt defaults to 4
```

**TAB CHARACTER vs TAB WIDTH — three separate settings, only one is in the file.**
Verified by `od -c` on both the rustfmt and the ESLint output: what lands on disk is one
literal `\t` per level (`\t i f`, `\t {`, `\t \t l e t`). Correct. But:

1. **Stored** — one real TAB per level. Set by `hard_tabs`/`UseTab`/`indent: 'tab'`. ✅
2. **Displayed** — 2 columns. An EDITOR setting (`.editorconfig` `indent_size = 2`);
   it never changes the bytes. A terminal showing 8 columns is the terminal, not the file.
3. **Assumed by the formatter** — each tool does line-length and alignment math against
   its OWN tab width, and they do not default to 2. If this is left wrong the tools wrap
   lines as though the indentation were 2-4x wider than it looks on screen:
   - rustfmt `tab_spaces = 2` — **defaults to 4**, was missing from the first draft
   - clang-format `TabWidth: 2` (with `UseTab: Always`)
   - ESLint `@stylistic/indent: ['error', 'tab']` — confirm its tab-length option in the
     Unit 6 config pass, together with the `.svelte` `<script>` offset item below

Run as `rustup run nightly rustfmt` (or `cargo +nightly fmt`). Nightly is installed
(`nightly-aarch64-apple-darwin`) and is used for FORMATTING ONLY — builds, tests and
both shipped wheels stay on stable, because formatting cannot change the compiled
artifact. §3.1's warning stands and is confirmed: on STABLE the same config warns and
silently emits K&R with exit 0.

**2. TypeScript: ESLint + `@stylistic` does what Prettier cannot.** §3.2 is correct that
Prettier can never emit Allman, and correct to reject dprint (no `.svelte` support) — but
it concluded from there that K&R was forced. It is not. `@stylistic/brace-style` has an
explicit `"allman"` option and is auto-fixable. Verified on a `.ts` and a `.svelte` file:

```ts
export function pick(pop: number[]): number
{
	if (pop.length > 0)
	{
		return pop[0]        // semicolon removed, tab-indented
	}
	else                     // NOT cuddled — real Allman
	{
		return -1
	}
}
```

Rules used: `@stylistic/brace-style: ['error','allman',{allowSingleLine:false}]`,
`@stylistic/semi: ['error','never']`, `@stylistic/indent: ['error','tab']`, with
`@typescript-eslint/parser` for `.ts` and `svelte-eslint-parser` for `.svelte`.

**The CSS-semicolon danger §3.3 identified does not apply to ESLint**: it does not touch
`<style>` blocks at all. All 3,042 CSS semicolons and 893 CSS braces were preserved
untouched in the test. Prettier is therefore NOT needed and should not be adopted.

⚠️ **One open config item:** inside `.svelte` `<script>` blocks the original 2-space base
offset survives and tabs begin after it, so those blocks come out mixed. `eslint-plugin-svelte`'s
own indent rule (installed) is the fix; it needs one config pass before Unit 6 runs.

**3. Installed 18/08/2026** (all were missing): `nightly` toolchain + rustfmt,
`clang-format` 21.1.5 (Homebrew), `ruff` 0.16.3 (project venv), and
`eslint` + `@stylistic/eslint-plugin` + `eslint-plugin-svelte` + `svelte-eslint-parser`
+ `@typescript-eslint/parser` (dashboard/frontend devDependencies).

**4. Unaffected by these corrections** — the survey's measurements stand: the file/line
inventory (§1-2), the "do NOT use rustfmt for the indentation migration, it costs
+11,312/−3,722 lines of reflow churn" finding, the `common.metal` prepend-site safety
proof, the 2 genuinely-mixed files, and the live-job safety ordering. Note that choosing
Allman for Rust means accepting that reflow churn ONCE, since Allman requires rustfmt.

---

## 0. Executive summary

Every number in this document was **measured**, not estimated. The measurement scripts
ran read-only against the working tree; the one write was this file.

The headline is that this migration is **much smaller than it looks**, because it is
really three unrelated jobs that have been conflated:

| Job | Scale | Feasibility |
|---|---|---|
| **Indentation → TAB** | ~30k lines, mostly in ONE crate | Fully automatable, provably safe |
| **Braces → Allman** | ~8.4k brace sites repo-wide | NOT automatable on stable Rust; impossible in Prettier |
| **Optional semicolons** | 208 Python + 2,684 TS/Svelte-script | Automatable, but CSS must be excluded |

**Python is essentially already compliant** (see §2) — 86% of `src/wnn` is tab-indented
and *zero* files are mixed. The single biggest indentation debt is the
`ram_accelerator` worker crate (41 of 50 files space-indented).

**The honest verdict on braces:** Allman is unreachable by configured formatter for Rust
(stable) and for TypeScript (Prettier, at all). Only Metal can be machine-Allman'd. §3
lays out the real options instead of proposing a hand-rolled rewriter.

---

## 1. Inventory (measured 17/08/2026)

### 1.1 Method

- **Python indentation** was classified with `tokenize` **INDENT tokens** — the real block
  indentation. This is immune to the false positives you get from grepping for leading
  spaces, because wrapped-argument continuation lines are aligned with spaces even in
  correctly tab-indented files (and the house style explicitly permits that alignment).
- **Brace languages** were classified by the first character of the indent run, counting
  only lines whose previous non-blank line ends a statement or block (`{ } ; : ) */ >`),
  again to exclude continuation alignment. A file is `MIXED` only when the minority
  indent character exceeds 5% of its block-indent lines.
- **Excluded** from all counts: `.git`, `node_modules`, `target`, `__pycache__`,
  `.venv`/`wnn-venv`/`site-packages`, `build`, `dist`, `dist_staged`, `.svelte-kit`.
- **Included** (a first pass wrongly pruned it): `src/wnn/ram/experiments/**`, which is
  real source, not experiment output.

### 1.2 The table

| Group | files | lines | TAB | SPACE | MIXED | none | K&R `{` | Allman `{` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **rust:** `ram_accelerator` (worker) | 50 | 34,669 | 7 | **41** | **1** | 1 | 3,461 | 54 |
| **rust:** `ram_core` | 8 | 5,879 | 3 | **5** | 0 | 0 | 663 | 5 |
| **rust:** `ram_controller` | 21 | 25,303 | **20** | 1 | 0 | 0 | 2,528 | 11 |
| **rust:** dashboard backend | 25 | 6,427 | 0 | **25** | 0 | 0 | 606 | 8 |
| **python:** `src/wnn` | 255 | 88,884 | **220** | 15 | **0** | 20 | – | – |
| **python:** `scripts` | 180 | 30,833 | **166** | 13 | **1** | 0 | – | – |
| **python:** `tests` | 123 | 23,811 | **100** | 20 | **1** | 2 | – | – |
| **metal:** shaders | 16 | 5,774 | 4 | **12** | 0 | 0 | 420 | 23 |
| **svelte:** dashboard | 45 | 14,168 | 6 | **39** | 0 | 0 | 1,409\* | 0 |
| **ts/js:** dashboard | 14 | 1,834 | 1 | **13** | 0 | 0 | 189 | 0 |
| **TOTAL** | **737** | **237,582** | 527 | 184 | 2 | 23 | ~8,366\* | 110 |

\* The Svelte brace count is inflated by CSS. Split measured separately — see §1.5.

Out of the named scope but measured while walking: `scripts/**/*.sh` — 186 files,
16,887 lines, 94 TAB / 31 SPACE / 12 MIXED / 49 no-indent. Shell is not covered by this
plan; flagged so it is not mistaken for "done".

### 1.3 MIXED-indentation files — the highest-risk set

Only **two** files in the entire repo are genuinely mixed, plus one marginal case:

| File | lines | tab indents | space indents | Verdict |
|---|---:|---:|---:|---|
| `src/wnn/ram/strategies/accelerator/bitwise_ramlm.rs` | 2,507 | 138 | 1,095 | **Real mix.** Predominantly space with a tab-indented island. Handle alone, in its own commit. |
| `scripts/build_oi_vs_old_report.py` | 429 | 23 | 46 | **Real mix, and it is Python** — indentation is syntax here. Highest risk file in the repo. |
| `tests/controller_holdout_threshold_alignment.py` | 118 | 10 | 4 | Marginal mix; small and test-only. |

That is the complete mixed-file list. Everything else is internally consistent, which is
what makes a mechanical migration viable.

### 1.4 Files needing indentation migration, by unit

**`ram_core` — 5 files, 3,292 space-indented lines**
`core/neighbor_search.rs` (2,118 L), `core/sparse_memory.rs` (1,398 L),
`core/metal_sparse.rs` (399 L), `core/packed_bits.rs` (347 L), `core/lib.rs` (52 L).
Already tab: `cancel.rs`, `neuron_memory.rs`, `counter_rng.rs`.

**`ram_controller` — 4 files, 281 space-indented lines**
Only `controller/lib.rs` (615 L) is space-indented as a whole; three other files carry a
handful of stray space-indent lines. **20 of 21 files are already compliant.**

**`ram_accelerator` (worker) — 42 files, 22,036 space-indented lines**
The bulk of the debt. Largest: `multistage.rs` (3,451 L), `adaptive/eval_hybrid.rs`
(1,397 L), `pyapi/tiered_sparse.rs` (1,377 L), `ids_cache.rs` (1,336 L),
`metal_genome_eval.rs` (1,264 L), `adaptive/eval_single.rs` (1,031 L), `ramlm.rs`
(1,006 L). Already tab: `metal_train.rs`, `metal_stats.rs`, `marker_probe.rs`,
`marker_train.rs`, `adaptation.rs`, `atomic_hashtable.rs`, `pyapi/mod.rs`.

**Metal — 12 files, 1,639 space-indented lines**
Including `core/shaders/common.metal` (167 L) and `core/shaders/sparse_forward.metal`
(517 L). Already tab: `bitwise_ce.metal`, `neuron_stats.metal`,
`controller/shaders/controller_rollout.metal` (2,694 L — the largest shader, already
compliant), `controller/shaders/controller_sep.metal`.

**Python — 50 files total across all three trees**
`src/wnn`: 15 files / 7,685 lines / 6,170 space-indented lines.
`scripts`: 14 files / 2,626 lines / 1,671 space-indented lines.
`tests`: 21 files / 2,851 lines / 1,878 space-indented lines.

The `src/wnn` set is worth naming because five of them are **live worker code**:
`ram/experiments/worker.py` (2,076 L), `ram/experiments/data_layer.py` (1,026 L),
`ram/experiments/tracker.py` (745 L), `ram/experiments/flow_runner.py` (363 L),
`ram/experiments/scheduler.py` (100 L). The rest: `ram/core/gating.py` (995 L),
`ram/architecture/tiered_evaluator.py` (815 L), `representations/*.py` (5 files),
`seeds.py`, `ram/architecture/genome_log.py`, `core/thresholds.py`.

**Notably: `src/wnn/control/**` — the live controller package — is 100% tab-indented
already.** It needs no indentation work at all.

### 1.5 Semicolons

| Surface | Count | Removable? |
|---|---:|---|
| Python `src/wnn` | 10 tokens in 4 files | Yes (`phased_ga.py` 6, `dagger.py` 2, `optimal.py` 1, `controller_grid_search.py` 1) |
| Python `scripts` | 141 tokens in 41 files | Yes |
| Python `tests` | 57 tokens in 18 files | Yes |
| Svelte `<script>` (TS) | 1,935 | Yes |
| Svelte `<style>` (CSS) | **3,042** | **NO — required CSS syntax** |
| Svelte markup | 45 | Mostly HTML entities; inspect individually |
| `.ts` / `.js` | 750 | Yes |
| Rust (all crates) | 15,908 | **NO — load-bearing** |
| Metal | 1,799 | **NO — required C++ syntax** |

Python semicolon counts are exact — measured as `OP` tokens with value `;` via
`tokenize`, so string and comment contents cannot inflate them.

The Svelte split matters: a naive "strip trailing semicolons from `.svelte`" would
destroy 3,042 lines of CSS. Likewise the 1,409 "K&R braces" in Svelte are really
**499 in `<script>` and 893 in `<style>`** — CSS rule braces that must not be moved.

---

## 2. What is already compliant

**Separate the two jobs — they have different risk profiles and different tooling.**

### Indentation: 527 of 737 files (71%) are already TAB

- **Python is effectively done.** 220/255 in `src/wnn`, 166/180 in `scripts`, 100/123 in
  `tests` are tab-indented, and **zero** `src/wnn` files are mixed. The remaining 50
  files are isolated, self-consistent space-indented files — no interleaving.
- **`ram_controller` is effectively done** — 20/21 files, 281 lines outstanding.
- **The real debt is concentrated**: `ram_accelerator` alone is 22,036 of the ~30k
  space-indented lines in scope, i.e. **73% of the whole indentation job is one crate**.

### Python: indentation + stray semicolons only, and it is nearly a no-op

Python has no braces, so the entire Allman question does not apply. What remains is:

1. 50 files to convert space → tab (§1.4), and
2. 208 stray semicolons to delete.

**Say it plainly: the Python "migration" is ~13,200 lines of pure whitespace across 50
files plus 208 one-character deletions.** There is no structural change of any kind.
That is the good news; §4(b) is the bad news about proving it.

### Braces: essentially nothing is compliant

110 Allman brace sites against ~8,366 K&R sites repo-wide — and many of those 110 are
incidental (a lone `{` from a wrapped expression), not deliberate style. Treat brace
migration as starting from zero.

---

## 3. Brace feasibility, per language

### 3.1 Rust — `brace_style` is nightly-only. Confirmed empirically, not from memory.

Probed on this machine, 17/08/2026:

```
$ rustfmt --version
rustfmt 1.8.0-stable (254b59607d 2026-01-19)
$ rustup toolchain list
stable-aarch64-apple-darwin (active, default)      # no nightly installed
```

With `brace_style = "AlwaysNextLine"` in `rustfmt.toml`, stable rustfmt prints:

```
Warning: can't set `brace_style = AlwaysNextLine`, unstable features are only
available in nightly channel.
```

…and **formats the file anyway with the brace on the same line**, exit code 0. This is
the worst failure mode: it is a warning, not an error, so a config carrying
`brace_style` would silently do nothing on every stable run while looking configured.

By contrast, `hard_tabs = true` **is stable** — verified by round-trip:

```
input:  fn main() {\n    let x = 1;\n}
output: fn main() {\n\tlet x = 1;\n}
```

So for Rust the two jobs cleanly separate: **indentation is machine-applicable on
stable today; braces are not.**

**Options, honestly stated:**

| Option | Cost | Risk | Recommendation |
|---|---|---|---|
| **A. Accept K&R for Rust.** Apply `hard_tabs` only; leave braces. | Zero | Zero | **Recommended.** Buys 100% of the indentation win for 0% of the brace risk. |
| **B. Pin a nightly toolchain for formatting only.** `rustup toolchain install nightly`, add `rust-toolchain.toml` *not* at repo root (it would change the build toolchain) — instead invoke `cargo +nightly fmt` explicitly from a `make fmt` target. | One toolchain install; a second toolchain to keep current | Moderate: nightly rustfmt output drifts between nightlies, so two developers on different nightlies produce different formatting. Also `brace_style` has been unstable for ~8 years with no stabilisation path. | Viable if Luiz wants real Allman. Pin the exact nightly date in the Makefile and in `.git-blame-ignore-revs` notes. |
| **C. Hand-maintain with a CI checker.** Write no rewriter; add a checker that flags `) {$` at end of line and fail the commit. | Low to build; high ongoing | High friction — every future Rust edit must be hand-Allman'd, and rustfmt would fight it if anyone ran it. | Not recommended alongside rustfmt; the two would oscillate. |
| **D. Hand-rolled Allman rewriter.** | — | — | **Rejected.** A regex/line-based brace mover cannot distinguish a block brace from a struct literal, a closure, a `match` arm, or a brace inside a string/macro. This is precisely the fragile hand-rolled rewriter the role brief warns against. |

**A caveat that must be settled before ANY rustfmt run, including option A:** rustfmt is
not an indentation tool, it is a *reformatter*. Measured on a scratch copy, running
`rustfmt` with `hard_tabs = true, tab_spaces = 2, max_width = 100`:

| Crate | rustfmt diff | indentation-only diff |
|---|---|---|
| `ram_core` | 7 files, +4,417 / −3,805 | **5 files, 3,292 lines** |
| `ram_controller` | **21 files, +11,312 / −3,722** | **4 files, 281 lines** |
| `ram_accelerator` | 49 files, +31,596 / −26,766 | 42 files, 22,036 lines |

Look at `ram_controller`: it is already tab-indented, so rustfmt's 15,000-line diff is
**pure line-rewrapping churn with zero style benefit**. Running `cargo fmt` wholesale
would be a catastrophe for review and for `git blame`.

**Therefore: do not use rustfmt for the migration.** Use a leading-whitespace-only
transform (§6.4), and adopt `rustfmt.toml` only as a *forward* convention for
newly-written code, applied per-file on request — never repo-wide.

### 3.2 TypeScript / Svelte — Prettier cannot produce Allman. At all.

Prettier has no brace-position option and has closed every request for one as
out-of-scope; its explicit design position is a single opinionated output. There is no
config, plugin hook, or flag that yields Allman for TS. **Stating that plainly: option
"configure Prettier for Allman" does not exist.**

Also measured: **Prettier is not installed and not configured in this repo.**
`dashboard/frontend/package.json` has no `prettier` dependency and there is no
`.prettierrc` anywhere. So there is no existing formatter to reconfigure — only a
decision about whether to introduce one.

**Options:**

| Option | Effect | Recommendation |
|---|---|---|
| **A. Introduce Prettier with `useTabs: true, semi: false`.** | Wins tabs + semicolon removal (2,684 sites) mechanically and permanently. **Loses Allman** — Prettier will enforce K&R forever. | **Recommended.** It delivers 2 of the 3 house rules on this surface, automatically. Accept K&R for TS, consistent with accepting it for Rust (option 3.1-A). |
| **B. dprint** (`typescript` plugin). | Has `braceposition` — but its values are `maintain`/`sameLine`/`nextLine`/`sameLineUnlessHanging`, and `nextLine` **does** give Allman for TS. | The only real Allman-for-TS path. Cost: a second, less common formatter in the stack, and it does **not** handle `.svelte` files — which is 45 of the 59 frontend files. Not worth it for 14 `.ts` files. |
| **C. No formatter; hand-migrate + checker.** | Same oscillation problem as 3.1-C. | Not recommended. |
| **D. Hand-rolled TS brace rewriter.** | — | **Rejected**, same reasoning as 3.1-D, and worse: `.svelte` files interleave TS, HTML and CSS in one file. |

**Hard constraint for any Svelte tooling:** the semicolon and brace rules apply to
`<script>` only. Prettier handles this correctly out of the box (it parses each block
with the right sub-parser and leaves CSS semicolons alone). A hand-rolled tool would
not. This is a decisive argument for Prettier over anything bespoke on this surface.

**Accessibility floor:** CLAUDE.md mandates `1rem` base font size for dashboard body
content. Formatting cannot violate this, but a `<style>`-block reformat makes the CSS
diff large enough to hide a regression. Verify with a grep for `font-size` values below
`1rem` after any Svelte commit.

### 3.3 Metal — the one surface where Allman is genuinely automatable

Metal is C++-like, so `clang-format` applies with `BreakBeforeBraces: Allman`,
`UseTab: Always`, `TabWidth: 2`, `IndentWidth: 2`.

`clang-format` is **not currently installed** (`which clang-format` → not found);
`brew install clang-format` is a prerequisite.

Two mechanical details, both measured:

1. **`.metal` is not a recognised extension.** clang-format picks its language from the
   filename and will not infer C++ from `.metal`. Invoke as
   `clang-format --assume-filename=shader.cpp < in.metal`, or set `Language: Cpp` and
   pass the files explicitly. Do not assume it "just works".
2. **`SortIncludes` must be `false`.** `common.metal` opens with
   `#include <metal_stdlib>` followed by `using namespace metal;`. Include reordering
   across the concatenated unit would be a real hazard.

**Is clang-format safe given that `common.metal` is prepended to every other shader?**
Measured — yes, and here is the evidence rather than an assurance:

- The prepend is a **compile-time Rust string concat**, at 11 sites, and every one of
  them uses an **explicit `"\n"` separator**:

  ```rust
  let shader_source = concat!(include_str!("core/shaders/common.metal"), "\n",
                              include_str!("shaders/ramlm.metal"));
  ```

  Sites: `metal_ramlm.rs:42`, `metal_genome_eval.rs:237,529,551`, `metal_train.rs:143`,
  `metal_stats.rs:94`, `marker_probe.rs:24`, `marker_train.rs:25`,
  `core/metal_sparse.rs:38`, `controller/metal_controller.rs:308,1441`.
  A second shared prelude, `core/shaders/marker_slots.metal`, is prepended the same way.

  Because the separator is explicit, the concatenation **cannot** be broken by a
  formatter adding or removing a trailing newline, and it cannot be broken by a trailing
  `//` comment on the last line of `common.metal`.

- **There are zero backslash line-continuations in any `.metal` file in the repo**
  (measured: `grep -c '\\$'` over all 16 shaders returns no non-zero file). So there is
  no multi-line macro for clang-format to reflow and break. This was the main residual
  hazard and it does not exist here.

- `common.metal` is prepended **first**, so its `#include` remains at the top of the
  translation unit regardless of formatting.

**Verdict: Metal is safe to clang-format to full Allman + tabs.** It is also small
(1,639 lines of indentation, 420 brace sites) and has a direct compile test.

### 3.4 Python

No braces. Nothing to decide. Indentation + 208 semicolons only.

One tooling warning: **do not reach for `black`.** Black cannot emit tabs — it is
hard-coded to 4 spaces and would revert the entire Python tree. If a Python formatter is
wanted, it must be **`ruff format` with `indent-style = "tab"`** (`ruff` is already
declared in the `dev` extra of `pyproject.toml`, though not currently installed).

---

## 4. Risk register

### (a) Rust semicolons are load-bearing — 15,908 of them

A trailing `;` turns an expression into a statement. Dropping one from the last
expression of a block changes the block's value from `T` to `()` — which usually fails
to compile, but **not always**: in a function returning `()`, in a `match` arm used for
effect, or where the value is inferred, it can compile and silently change behaviour.

**Controls:**
- The house rule explicitly exempts Rust/C/C++/C#/Java from semicolon removal. No tool
  in this plan is permitted to touch a Rust `;`.
- The transform in §6.4 operates on **leading whitespace only** and cannot reach a
  semicolon by construction.
- Verification is by compile + full test run, not inspection.

### (b) Python indentation IS syntax

A tab/space change in Python is a semantic change until proven otherwise. Worse, Python
resolves tabs to the *next multiple of 8* when comparing indent levels, so a file that
mixes them can change block structure under an apparently innocent edit — this is
exactly why `scripts/build_oi_vs_old_report.py` (23 tab / 46 space indents) is the
highest-risk file in the repo.

**Controls — mandatory, in this order:**
1. `python -m compileall -q <paths>` — proves it still parses.
2. `python -m tokenize` round-trip: the token stream (types + strings, ignoring
   `INDENT`/`DEDENT` whitespace) must be **byte-identical** before and after. This is a
   much stronger proof than compileall: it proves no statement moved between blocks.
3. `python -X dev -W error::SyntaxWarning -c "import ..."` for the touched modules.
4. `pytest` on the affected tree.
5. **Never by eye.** A reviewer cannot see the difference between a tab and 4 spaces in
   a diff view, which is precisely why this rule exists.

Additionally: Python 3 **rejects** inconsistent tab/space use with `TabError` in most
cases, which converts the majority of failure modes into loud parse errors. Rely on
that, but do not rely on it alone — the residual cases are the dangerous ones.

### (c) `common.metal` is prepended at compile time

Covered in detail in §3.3. Summary of the measured position: 11 concat sites, all with
an explicit `"\n"` separator, zero backslash continuations repo-wide, `#include` stays
first. **Risk assessed as LOW**, conditional on `SortIncludes: false`.

**Control:** after any shader commit, run the two parity/compile tests named in
`common.metal`'s own header comment:
`cargo test metal_shaders_compile` and `cargo test cpu_fallback_matches_gpu`.
A shader that fails to compile is caught at library-load time, i.e. immediately.

### (d) Style churn destroys `git blame`

~30k lines of whitespace change will bury authorship for every line it touches.

**Control — non-negotiable, and set up BEFORE the first style commit:**

1. Create `.git-blame-ignore-revs` at repo root.
2. After **each** style-only commit, append its full 40-char SHA with a comment line.
3. Configure the repo so it is used by default:
   ```bash
   git config blame.ignoreRevsFile .git-blame-ignore-revs
   ```
   (GitHub honours this file automatically; the `git config` line is for local `git blame`.)
4. Every style commit message must begin with `style:` and state
   `Formatting only — no behavior change.` plus the verification command that was run
   and its result. This makes the commits trivially skippable during `git bisect`.

**Additional control:** keep style commits **strictly** separate from functional commits.
Never mix. A style commit that also fixes a bug is unbisectable and unreviewable.

### (e) The live jobs — the scheduling risk

Verified live at time of survey (17/08/2026 21:47 ET):

| PID | What | Constraint |
|---|---|---|
| **2386** | `bash scripts/recalc_then_resume_ladder.sh` → `scripts/sweep_ladder_chain.sh` | Started 17/08 21:47. **Never edit a `.sh` while bash is executing it** — bash resumes at a byte offset. |
| **4976** | `python -m wnn.control.phased_ga --recalc-headline …` | A **fresh process per run**, lazily importing `src/wnn/**`. Editing Python under `src/wnn/` mid-run corrupts or kills a ~4h run. |
| **32819** | `python -m wnn.ram.experiments.worker` (IDS worker) | Imports `src/wnn/ram/experiments/**` and the `ram_accelerator` wheel. |

Derived rules for this migration:

- **Rust and Metal source may be edited at any time.** Live processes execute the
  *installed wheels*, not the source tree. Editing `.rs`/`.metal` cannot affect a
  running process.
- **Installing a wheel is gated**, not editing it. `ram_controller` is loaded by the live
  chain: per the standing rule *never deploy while a chain is armed*, defer the install
  to a chain boundary. `ram_accelerator` may only be swapped at worker-idle via
  `scripts/worker_swap.py`.
- **`src/wnn/**/*.py` requires a chain-idle window.** No exceptions.
- **`scripts/*.py` and `scripts/*.sh`** are unsafe while the chain runs — the chain
  invokes them.
- **`tests/**` and `dashboard/**` are safe any time**, with one caveat: if `vite dev` is
  running, editing `.svelte` triggers an HMR rebuild that briefly disturbs the dashboard
  you are using to watch the chain. Do it when you do not need the dashboard.
- **Source/wheel skew:** a style-only Rust commit that is not immediately built leaves
  the tree ahead of the installed wheel. For a *provably* behaviour-neutral change this
  is acceptable, but it must be called out in the commit message and the wheel rebuilt
  at the next natural boundary. Do not let skew accumulate across several units.

### (f) A `ram_core` change rebuilds BOTH wheels

`ram_core` is the shared substrate. Touching it forces a rebuild of `ram_accelerator`
*and* `ram_controller`, which means a **worker swap at idle** — the expensive
deployment. Schedule `ram_core` deliberately, not opportunistically. This is why it is
not the first unit despite being small.

---

## 5. Ordered work plan

Each unit is one commit, style-only, independently reviewable and revertable.

Legend for **Safe when**:
🟢 any time · 🟡 needs chain boundary (install only) · 🔴 needs chain + worker idle

---

### Unit 0 — Automation configs *(no source files touched)*

| | |
|---|---|
| **What changes** | Add `.editorconfig`, `rustfmt.toml`, `.clang-format`, `.git-blame-ignore-revs` (empty, with header), `dashboard/frontend/.prettierrc` + devDependency, and `scripts/check_style.py`. |
| **Diff size** | ~120 lines, all new files. Zero existing files modified. |
| **Verification** | `git diff --stat` shows only new files. `python scripts/check_style.py --report` runs and prints the current violation counts (should reproduce §1's table). |
| **Safe when** | 🟢 **Any time.** Touches no source and no live path. |

**This is the correct first commit.** It changes nothing, but it makes every later unit
mechanical and reviewable, and it establishes `.git-blame-ignore-revs` *before* the
first noisy commit rather than after.

---

### Unit 1 — `ram_controller` indentation → TAB

| | |
|---|---|
| **What changes** | 4 files, 281 lines. Effectively just `controller/lib.rs` (615 L) plus stray lines in 3 others. Leading whitespace only. No braces, no semicolons. |
| **Diff size** | **281 lines** — the smallest meaningful unit in the plan. |
| **Verification** | `PYO3_PYTHON="/Volumes/20260401-WDBlack-SN850X-2TB/wnn/venv/bin/python" cargo test -p ram_controller --lib --no-default-features` — 171 `#[test]` attributes present in the crate (CLAUDE.md cites 94 executing under this configuration, including the 14 CPU/GPU parity sweeps; reconcile the count on first run and correct whichever doc is stale). Then `cargo check --workspace`. |
| **Safe when** | 🟢 to edit, commit and test. 🟡 to *install* the rebuilt wheel — defer `maturin develop --release -m controller/Cargo.toml` to a chain boundary. |

**Recommended first source unit.** Smallest diff, strongest test coverage in the repo,
and the cheapest deployment (controller wheel needs no worker swap). It is the ideal
rehearsal for the transform and the verification loop before anything larger is attempted.

---

### Unit 2 — Metal shaders → TAB **+ Allman** (the full house style)

| | |
|---|---|
| **What changes** | 12 files, 1,639 indentation lines + 420 brace sites moved to their own line. `clang-format` with `BreakBeforeBraces: Allman`, `UseTab: Always`, `SortIncludes: false`. |
| **Diff size** | ~2,000–2,400 changed lines. |
| **Verification** | `cargo test metal_shaders_compile` and `cargo test cpu_fallback_matches_gpu` (both named in `common.metal`'s own header). Shader compile failures surface at library load, so this is a sharp test. |
| **Safe when** | 🟢 to edit/commit/test. 🟡 to install (shaders are `include_str!`-embedded, so they ship inside the wheels). |
| **Prerequisite** | `brew install clang-format`. |

Worth doing early: it is the **only** surface that reaches the complete house style, so
it validates the Allman end state on real code at low cost.

---

### Unit 3 — `ram_accelerator` (worker) indentation → TAB, part 1: `pyapi/` + `adaptive/`

| | |
|---|---|
| **What changes** | ~26 files. `pyapi/` (16 files, ~6,000 L) and `adaptive/` (12 files, ~7,300 L). Leading whitespace only. |
| **Diff size** | ~9,500 lines. |
| **Verification** | `cargo check --workspace`; `cargo test -p ram_accelerator --lib --no-default-features`. |
| **Safe when** | 🟢 to edit/commit/test. 🔴 to install — worker wheel, `scripts/worker_swap.py` at worker-idle only. |

Split from Unit 4 purely to keep the reviewable diff under ~10k lines.

---

### Unit 4 — `ram_accelerator` indentation → TAB, part 2: crate root

| | |
|---|---|
| **What changes** | ~16 remaining root files: `multistage.rs` (3,451 L), `ids_cache.rs`, `metal_genome_eval.rs`, `ramlm.rs`, `multiclass_metrics.rs`, `token_cache.rs`, `ids_streaming.rs`, `gating.rs`, `metal_gating.rs`, `lib.rs`, `eval_worker.rs`, `metal_evaluator.rs`, `metal_ramlm.rs`. |
| **Diff size** | ~12,500 lines. |
| **Verification** | Same as Unit 3. |
| **Safe when** | 🟢 edit/commit/test · 🔴 install. |

---

### Unit 5 — `bitwise_ramlm.rs` alone *(the mixed-indentation file)*

| | |
|---|---|
| **What changes** | One file, 2,507 lines, 138 tab + 1,095 space indents. |
| **Diff size** | ~1,100 lines. |
| **Verification** | Same as Unit 3, plus a manual read of the tab-indented island to confirm nesting depth is preserved. |
| **Safe when** | 🟢 edit/commit/test · 🔴 install. |

**Deliberately isolated.** Mixed files are where a mechanical transform is most likely to
change nesting depth. Give it its own commit so a revert is surgical.

---

### Unit 6 — `ram_core` indentation → TAB

| | |
|---|---|
| **What changes** | 5 files, 3,292 lines: `neighbor_search.rs`, `sparse_memory.rs`, `metal_sparse.rs`, `packed_bits.rs`, `lib.rs`. |
| **Diff size** | ~3,300 lines. |
| **Verification** | `cargo check --workspace`; `cargo test -p ram_core --lib`; then **both** dependent suites: `cargo test -p ram_controller --lib --no-default-features` and `cargo test -p ram_accelerator --lib --no-default-features`. |
| **Safe when** | 🟢 edit/commit/test · 🔴 install — **forces a rebuild of BOTH wheels**, so the worker swap is mandatory (§4f). |

Sequenced here, after the crates that depend on it are already converted, so the
both-wheels rebuild happens exactly once and can be bundled with Units 3–5's install.

---

### Unit 7 — `tests/**` Python: indentation + semicolons

| | |
|---|---|
| **What changes** | 21 files, 1,878 space-indented lines; 57 stray semicolons in 18 files. Includes the marginal mixed file `controller_holdout_threshold_alignment.py`. |
| **Diff size** | ~1,950 lines. |
| **Verification** | `python -m compileall -q tests/`; tokenize round-trip equality (§4b control 2); `pytest tests/ -x -q`. |
| **Safe when** | 🟢 **Any time.** Nothing live imports `tests/`. |

Sequenced before the riskier Python units as a rehearsal of the Python verification loop
on a surface where a mistake costs nothing.

---

### Unit 8 — `src/wnn/**` Python, part A: non-live packages

| | |
|---|---|
| **What changes** | 10 files: `ram/core/gating.py` (995 L), `ram/architecture/tiered_evaluator.py` (815 L), `representations/*.py` (5 files, 1,154 L), `seeds.py`, `ram/architecture/genome_log.py`, `core/thresholds.py`. Plus the 10 stray semicolons in `control/` (`phased_ga.py` ×6, `dagger.py` ×2, `optimal.py`, `controller_grid_search.py`). |
| **Diff size** | ~3,400 lines. |
| **Verification** | `python -m compileall -q src/wnn/`; tokenize round-trip; `pytest tests/ -q`; import-smoke each touched module. |
| **Safe when** | 🔴 **Chain idle required.** `phased_ga.py` is imported by the live controller, and `gating.py` / `tiered_evaluator.py` are reachable from worker paths. |

**Note the trap:** the semicolon edits here touch `src/wnn/control/phased_ga.py`, which
the live chain launches per run. Even a one-character edit mid-run risks a partially
written file being imported. This unit waits for a real idle window.

---

### Unit 9 — `src/wnn/ram/experiments/**`: the live worker package

| | |
|---|---|
| **What changes** | 5 files, ~4,300 lines: `worker.py` (2,076 L), `data_layer.py` (1,026 L), `tracker.py` (745 L), `flow_runner.py` (363 L), `scheduler.py` (100 L). |
| **Diff size** | ~3,300 lines. |
| **Verification** | `python -m compileall -q src/wnn/ram/experiments/`; tokenize round-trip; `pytest tests/ -q`; then a **live smoke**: start the worker and confirm it admits and completes one small flow before trusting it. |
| **Safe when** | 🔴 **Worker stopped.** This is the IDS worker's own source. Stop the worker, migrate, restart, smoke-test. Per the standing rule, only *queued* flows may be interrupted — never a running one. |

Highest operational risk unit in the plan. Do it alone, on a quiet day, not bundled.

---

### Unit 10 — `scripts/**` Python: indentation + semicolons

| | |
|---|---|
| **What changes** | 14 files, 1,671 space-indented lines; 141 semicolons in 41 files. Includes `build_oi_vs_old_report.py` — **the highest-risk file in the repo** (§1.3). |
| **Diff size** | ~1,850 lines. |
| **Verification** | `python -m compileall -q scripts/`; tokenize round-trip; execute each touched reporting script and diff its output against a pre-migration capture. |
| **Safe when** | 🔴 **Chain idle.** The chain invokes `scripts/`. |

**Split `build_oi_vs_old_report.py` into its own commit** within this unit — it is the
only genuinely mixed Python file, and its output is diffable, which gives a real
behavioural check rather than a parse check.

---

### Unit 11 — `dashboard/frontend`: tabs + semicolon removal (K&R retained)

| | |
|---|---|
| **What changes** | 59 files (45 `.svelte`, 14 `.ts`/`.js`). Prettier with `useTabs: true, semi: false`. Removes 1,935 `<script>` + 750 `.ts` semicolons; **leaves all 3,042 CSS semicolons and 893 CSS braces untouched**. Braces remain K&R (§3.2). |
| **Diff size** | ~12,000 lines (Prettier reflows). |
| **Verification** | `npm run check` (`svelte-check`); `npm run build`; load the dashboard and confirm the flows list, an experiment page and a chart all render. Grep for `font-size` below `1rem` to protect the accessibility floor (§3.2). |
| **Safe when** | 🟢 **Any time**, but prefer a window where the dashboard is not needed to watch the chain — HMR will churn while formatting. |

---

### Unit 12 — `dashboard/backend` (Rust) indentation → TAB

| | |
|---|---|
| **What changes** | 25 files, 4,793 space-indented lines. Separate crate (`wnn-dashboard`, edition 2024), not part of the accelerator workspace. |
| **Diff size** | ~4,800 lines. |
| **Verification** | `cargo check` / `cargo test` in `dashboard/`; restart the dashboard binary and confirm the API serves. |
| **Safe when** | 🟢 to edit/build. Restarting the dashboard is safe — it does not touch the chain or the worker. |

---

### Deferred / not in scope

- **`scripts/**/*.sh`** — 186 files, 12 of them mixed. Add as Unit 13 if wanted, but
  **only when the chain is fully idle**, because bash reads running scripts by byte
  offset.
- **Allman for Rust** — blocked on the §3.1 option A/B decision. If Luiz chooses option B
  (nightly), it becomes Units 14+ and should follow the same crate-by-crate split, at
  roughly **3,461 / 2,528 / 663 / 606 brace sites** per crate — i.e. brace migration is
  **10–20× the churn of the indentation work** it follows.

---

## 6. Automation

### 6.1 `.editorconfig` (repo root) — the foundation

```ini
root = true

[*]
indent_style = tab
indent_size = 2
tab_width = 2
end_of_line = lf
charset = utf-8
trim_trailing_whitespace = true
insert_final_newline = true

# Python: tabs, displayed at 2.
[*.py]
indent_style = tab
indent_size = 2

[*.{rs,metal}]
indent_style = tab
indent_size = 2

[*.{ts,js,svelte,css}]
indent_style = tab
indent_size = 2

# Files whose format is dictated by their ecosystem — do not fight them.
[*.{yml,yaml}]
indent_style = space
indent_size = 2

[*.md]
trim_trailing_whitespace = false

[Makefile]
indent_style = tab
```

`indent_size = 2` with `indent_style = tab` is exactly the house rule: a literal TAB
that *displays* as 2 columns.

### 6.2 `rustfmt.toml` (accelerator workspace root) — forward convention only

```toml
# House style: literal tabs displayed at 2 columns.
#
# NOTE: brace_style = "AlwaysNextLine" (Allman) is NIGHTLY-ONLY. On stable rustfmt it
# emits a warning and is SILENTLY IGNORED, so it is deliberately NOT set here — a
# setting that looks configured but does nothing is worse than no setting.
# See docs/code_style_migration_plan.md §3.1 for the Allman decision.
#
# Do NOT run `cargo fmt` across the workspace: measured, it produces ~15k lines of
# pure re-wrapping churn on ram_controller alone, which is already tab-indented.
# Use it per-file on code you are already editing.
hard_tabs  = true
tab_spaces = 2
max_width  = 100
newline_style = "Unix"
```

### 6.3 `.clang-format` (accelerator root, for `.metal`)

```yaml
---
Language: Cpp
BasedOnStyle: LLVM
BreakBeforeBraces: Allman
UseTab: Always
TabWidth: 2
IndentWidth: 2
ColumnLimit: 100
SortIncludes: false        # common.metal is PREPENDED to every shader — never reorder
AllowShortFunctionsOnASingleLine: None
AllowShortIfStatementsOnASingleLine: false
AllowShortLoopsOnASingleLine: false
PointerAlignment: Left
```

Invoke with an explicit filename assumption, because `.metal` is not a language
clang-format recognises:

```bash
clang-format --assume-filename=shader.cpp -i **/shaders/*.metal
```

### 6.4 The indentation transform — use this, not rustfmt

For Rust, Python and TS the migration is a **leading-whitespace-only** rewrite. It must
be string-literal aware. The measured hazard:

- **8 multi-line raw string literals repo-wide contain space-indented content lines**:
  `adaptive/validation.rs` (7 lines), `pyapi/ids_streamer_api.rs` (4),
  `metal_genome_eval.rs` (×2), `multistage.rs`, `controller/cpu_score.rs`,
  `controller/controller.rs`, `controller/cell_mode.rs` (1 each).
  In those, leading spaces are **data**, not indentation.

**Requirements for the transform:**
1. Convert leading runs of 4 spaces → 1 tab, at line start only. (Measured: only 19
   lines repo-wide have a non-multiple-of-4 indent, so residue is negligible — but
   preserve the remainder as spaces rather than rounding.)
2. **Skip any line inside a string literal** — for Rust, drive it from a real lexer
   (`proc-macro2` token spans) rather than a regex. For Python, use `tokenize` and skip
   `STRING` token ranges.
3. **Prove it**: assert that for every line, `old.lstrip() == new.lstrip()` and the line
   count is unchanged. Combined with (2), that is a complete proof the change is
   whitespace-only outside of string data.
4. For Python additionally assert token-stream equality (§4b).

That property — *"leading whitespace is the only thing that differs"* — is mechanically
checkable and is what makes these commits reviewable despite their size. Reviewers should
be told to verify the assertion output, not to read 22,000 lines.

Reviewers should also use `git diff -w` (ignore whitespace), which for a correct
indentation commit renders an **empty diff**. That is the single best review tool here.

### 6.5 Prettier (`dashboard/frontend/.prettierrc`)

```json
{
  "useTabs": true,
  "tabWidth": 2,
  "semi": false,
  "singleQuote": false,
  "printWidth": 100,
  "plugins": ["prettier-plugin-svelte"],
  "overrides": [
    { "files": "*.svelte", "options": { "parser": "svelte" } }
  ]
}
```

Add `prettier` and `prettier-plugin-svelte` to `devDependencies`, plus a script:

```json
"format": "prettier --write \"src/**/*.{ts,js,svelte,css}\"",
"format:check": "prettier --check \"src/**/*.{ts,js,svelte,css}\""
```

`semi: false` correctly leaves CSS semicolons alone — Prettier parses each Svelte block
with the appropriate sub-parser. Confirm this on the first file before running it across
all 45.

### 6.6 Pre-commit checker — `scripts/check_style.py`

A checker, not a fixer. Fixing on commit hides churn inside functional commits, which is
exactly what §4d forbids.

```
Usage:
  python scripts/check_style.py --staged     # pre-commit hook mode
  python scripts/check_style.py --report     # whole-repo violation counts

Checks, per file type:
  ALL      leading whitespace contains no space-run of >= 4 at line start
           (i.e. block indentation is tabs)
  *.py     no `;` OP token (tokenize-based, so strings/comments cannot false-positive)
  *.py     tokenize round-trip parses
  *.ts/js  no line ends with `;`
  *.svelte no line ends with `;` INSIDE <script> only  (CSS is exempt — §1.5)
  *.rs     (informational only) count of `) {` line endings, for the Allman backlog.
           NEVER fails the commit on this while option 3.1-A stands.
  *.metal  brace on own line (once Unit 2 has landed)

Exit 1 on violation, printing file:line. Bypass with `git commit --no-verify` for
in-flight work.
```

Install as `.git/hooks/pre-commit` (or via `pre-commit` framework). Note the repo
currently has **no** custom git hooks and only one GitHub workflow (`publish.yml`), so
this is new infrastructure — keep it dependency-free (stdlib only) so it never blocks a
commit for environment reasons.

### 6.7 `.git-blame-ignore-revs`

```
# Style-only commits. See docs/code_style_migration_plan.md.
# Enable locally with:
#   git config blame.ignoreRevsFile .git-blame-ignore-revs
#
# <sha>  # Unit 1: ram_controller indentation -> tab
# <sha>  # Unit 2: metal shaders -> tab + Allman
# ...
```

Append one line per style commit, immediately after making it.

---

## 7. Recommended decisions for Luiz

1. **Rust braces: accept K&R (option 3.1-A).** Take the indentation win now. Revisit
   Allman only if you are willing to pin a nightly toolchain for `fmt` — and note the
   brace job is 10–20× the churn of the indentation job it would follow.
2. **TS braces: accept K&R, adopt Prettier (option 3.2-A)** for tabs + semicolon
   removal. dprint is the only Allman-capable path and it does not cover `.svelte`,
   which is 45 of 59 files.
3. **Metal: go to full house style** — it is the one surface where everything is
   achievable and testable.
4. **Do not run `cargo fmt` repo-wide.** Measured, it is 15k lines of churn on an
   already-compliant crate.
5. **Start with Unit 0, then Unit 1.**

---

## 8. Appendix — reproducing these measurements

Measurement scripts were written to the session scratchpad, not to the repo. The
classification rules are specified in §1.1 in enough detail to reimplement. Key commands
used for the toolchain facts:

```bash
rustfmt --version                     # 1.8.0-stable
rustup toolchain list                 # stable only; no nightly
rustfmt --print-config default        # hard_tabs=false, brace_style="SameLineWhere"
which clang-format                    # not installed
grep -rn "include_str!" src/wnn/ram/strategies/accelerator --include=*.rs   # 11 concat sites
```

The `brace_style` finding was confirmed by round-trip on a probe file, not read from
documentation: stable rustfmt warns and formats with the brace on the same line anyway.
