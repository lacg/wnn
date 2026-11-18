#!/usr/bin/env bash
# CTRL-58 install chain (Luiz 18/11 ~09:25 EST): the n10 block's exact end line → box quiet →
# merge main into ctrl58-staged-fitness + cargo test → determinism gate (flag-off ABI 59 ×2 vs
# ABI 58) → Python suites on the installed 59 → merge into main + push. NO flight is launched
# (the A1/A2 flights wait for Luiz). Every step is a "CTRL58 CHAIN:" line in runs.log; every
# failure REFUSES and restores the ABI-58 wheel if it had been replaced.
set -u
R=/Users/lacg/wnn
WT=/Users/lacg/wnn-ctrl58
BR=ctrl58-staged-fitness
L=$R/logs/controller/ctrl48
RUNS=$L/runs.log
VENV=/Volumes/20260401-WDBlack-SN850X-2TB/wnn/venv
SITE=$VENV/lib/python3.13/site-packages
KEEP58=/Users/lacg/wnn-abi58-wheel-keep
G=$R/logs/controller/detgate
# The n10 driver's end line, EXACTLY ("<ISO timestamp> N10 DONE") — never a bare substring
# (17/11: a chain's own "waiting for ..." line false-triggered another watcher).
END_RE='^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9:]{8}Z N10 DONE$'
say() { local m="$*"; m=${m//N10 DONE/N10-END}; m=${m//DMY DONE/DMY-END}; m=${m//FF DONE/FF-END}; echo "$(date -u +%FT%TZ) CTRL58 CHAIN: $m" >> "$RUNS"; echo "[ctrl58-chain] $(TZ=America/New_York date '+%d/%m %H:%M %Z') $m"; }
fail() { say "REFUSED: $*"; exit 1; }
abi() { "$VENV/bin/python" -c "import ram_controller as c; print(c.ABI_VERSION)" 2>/dev/null; }
restore58() {
	if [ -d "$KEEP58/ram_controller" ]; then
		rm -rf "$SITE/ram_controller" "$SITE"/ram_controller-*.dist-info
		cp -a "$KEEP58"/ram_controller "$KEEP58"/ram_controller-*.dist-info "$SITE"/
		say "restored the ABI-58 wheel (now ABI $(abi))"
	fi
}

say "armed (pid $$): waiting for the n10 block's end line; then install ABI 59 + gate + merge (no flights)"
until grep -qE "$END_RE" "$RUNS"; do sleep 120; done
say "n10 block end line seen"
sleep 60
[ -f "$R/.git/MERGE_HEAD" ] && fail "the main tree is mid-merge — not touching it"
pgrep -f "wnn.control.phased_g[a]" >/dev/null && fail "a phased_ga is running"
[ "$(abi)" = 58 ] || fail "installed wheel is ABI $(abi), expected 58"

MAINBR=$(git -C "$R" rev-parse --abbrev-ref HEAD)
PRE=$(git -C "$R" rev-parse HEAD)
git -C "$WT" merge --no-edit "$MAINBR" >/dev/null 2>&1 || { git -C "$WT" merge --abort >/dev/null 2>&1; fail "merging $MAINBR ($PRE) into $BR conflicts"; }
say "merged $MAINBR ($PRE) into $BR: $(git -C "$WT" log --oneline -1 | cut -c1-90)"
( cd "$WT/src/wnn/ram/strategies/accelerator" && PYO3_PYTHON=$VENV/bin/python nice -n 5 cargo test -p ram_controller --lib --no-default-features ) > "$WT/chain_cargo_test.log" 2>&1 \
	|| fail "cargo test failed on the merged $BR (see $WT/chain_cargo_test.log)"
say "cargo test: $(grep 'test result' "$WT/chain_cargo_test.log" | tail -1)"

# --- determinism gate: OLD = main tree on the installed ABI 58, NEW1/NEW2 = merged branch on 59 ---
rm -rf "$KEEP58" && mkdir -p "$KEEP58" && cp -a "$SITE"/ram_controller "$SITE"/ram_controller-*.dist-info "$KEEP58"/ || fail "cannot keep a copy of the ABI-58 wheel"
GATE="$WT/scripts/ctrl_determinism_gate.sh"
rm -rf "$G/c58_abi58" "$G/c58_abi59a" "$G/c58_abi59b"
bash "$GATE" fly "$G/c58_abi58" "$R/src" >> "$G/c58_chain_gate.log" 2>&1 || fail "gate leg ABI 58 failed (see $G/c58_chain_gate.log)"
say "gate leg ABI 58 done"
( cd "$R" && unset CONDA_PREFIX && source wnn/bin/activate && cd "$WT/src/wnn/ram/strategies/accelerator" && maturin develop --release -m controller/Cargo.toml ) >> "$G/c58_chain_gate.log" 2>&1 \
	|| { restore58; fail "maturin develop (ABI 59) failed"; }
[ "$(abi)" = 59 ] || { restore58; fail "ABI 59 did not install (got $(abi))"; }
say "ABI 59 installed"
for r in c58_abi59a c58_abi59b; do
	bash "$GATE" fly "$G/$r" "$WT/src" >> "$G/c58_chain_gate.log" 2>&1 || { restore58; fail "gate leg $r failed"; }
done
if GATE_SRC=$WT/src bash "$GATE" compare "$G/c58_abi59a" "$G/c58_abi59b" "$G/c58_abi58" >> "$G/c58_chain_gate.log" 2>&1; then
	say "determinism gate PASS (flag-off ABI 59 x2 vs ABI 58)"
else
	restore58; fail "determinism gate FAIL — ABI 58 restored, nothing merged (see $G/c58_chain_gate.log)"
fi

# --- Python suites on the installed 59 (the refine suite + the real-scorer tail parity) ---
( cd "$WT/tests" && PYTHONPATH=$WT/src $VENV/bin/python -m pytest -q -p no:cacheprovider test_ctrl58_refine_fitness.py ) > "$G/c58_chain_pytest.log" 2>&1 \
	|| { restore58; fail "test_ctrl58_refine_fitness failed on ABI 59 — ABI 58 restored (see $G/c58_chain_pytest.log)"; }
( cd "$R" && PYTHONPATH=$WT/src $VENV/bin/python "$WT/tests/ctrl58_tail_scorer_parity.py" ) >> "$G/c58_chain_pytest.log" 2>&1 \
	|| { restore58; fail "ctrl58_tail_scorer_parity failed on ABI 59 — ABI 58 restored"; }
say "python: $(grep -E 'passed' "$G/c58_chain_pytest.log" | tail -1); tail scorer parity ALL PASS"

# --- merge + push (no flight) ---
git -C "$R" merge --ff-only "$BR" >/dev/null 2>&1 || git -C "$R" merge --no-edit "$BR" >/dev/null 2>&1 \
	|| { git -C "$R" merge --abort >/dev/null 2>&1; restore58; fail "merging $BR into $MAINBR failed — ABI 58 restored"; }
say "merged $BR into $MAINBR: $(git -C "$R" log --oneline -1 | cut -c1-90)"
PYTHONPATH=$R/src $VENV/bin/python -c "import ram_controller as c, wnn.control._accel as a; assert c.ABI_VERSION == 59 == a.EXPECTED_ABI" \
	|| fail "main tree facade / wheel are not both ABI 59 after the merge"
git -C "$R" push -q origin "$MAINBR" "$BR" >/dev/null 2>&1 && say "pushed $MAINBR and $BR" || say "WARNING: push failed (merge is local)"
say "done — ABI 59 installed and merged; no flight launched (A1/A2 wait for Luiz)"
