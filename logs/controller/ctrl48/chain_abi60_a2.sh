#!/usr/bin/env bash
# CTRL-58 A2 chain (Luiz 20/11 ~19:20 EST: "close Y3+L at n=5 and go to A2"). After the Y3+L block's exact end line:
# box quiet → merge this branch into ctrl60-mid-window + cargo test ram_controller → determinism gate (flag-off
# ABI 59 vs ABI 60 ×2, base = Y+F+L s1 argv) → Python suites on 60 → merge into the main tree + push →
# refine smoke (gate-size run with the A2 flags) → exec ab_a2.sh (A2 s1-s5). Every step is a "A2 CHAIN:" line in
# runs.log; any failure REFUSES and restores the ABI-59 wheel (nothing flies).
set -u
R=/Users/lacg/wnn
WT=/Users/lacg/wnn/.claude/worktrees/agent-ab1556f76be012006
BR=ctrl60-mid-window
L=$R/logs/controller/ctrl48
RUNS=$L/runs.log
VENV=/Volumes/20260401-WDBlack-SN850X-2TB/wnn/venv
SITE=$VENV/lib/python3.13/site-packages
KEEP59=/Users/lacg/wnn-abi59-wheel-keep
G=$R/logs/controller/detgate
BASE_OUT=$L/C-S2-d_wposx2_a59_rf_c200_dc_biasin_ms01_P_both_L1_ffF_yld_full_s1.out
EXTRA="--refine-fitness --refine-w-steady 0.40 --refine-w-mid 0.25 --refine-w-reward 0.10"
END_RE='^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9:]{8}Z Y3L DONE$'
say() { local m="$*"; m=${m//Y3L DONE/Y3L-END}; m=${m//A2 DONE/A2-END}; echo "$(date -u +%FT%TZ) A2 CHAIN: $m" >> "$RUNS"; echo "[a2-chain] $(TZ=America/New_York date '+%d/%m %H:%M %Z') $m"; }
fail() { say "REFUSED: $*"; exit 1; }
abi() { "$VENV/bin/python" -c "import ram_controller as c; print(c.ABI_VERSION)" 2>/dev/null; }
restore59() {
	if [ -d "$KEEP59/ram_controller" ]; then
		rm -rf "$SITE/ram_controller" "$SITE"/ram_controller-*.dist-info
		cp -a "$KEEP59"/ram_controller "$KEEP59"/ram_controller-*.dist-info "$SITE"/
		say "restored the ABI-59 wheel (now ABI $(abi))"
	fi
}

say "armed (pid $$): waits for the Y3+L block's end line, then ABI 60 install + gate + smoke, then A2 s1-s5"
until grep -qE "$END_RE" "$RUNS"; do sleep 60; done
say "Y3+L block end line seen"
sleep 60
[ -f "$R/.git/MERGE_HEAD" ] && fail "the main tree is mid-merge — not touching it"
pgrep -f "wnn.control.phased_g[a]" >/dev/null && fail "a phased_ga is running"
[ "$(abi)" = 59 ] || fail "installed wheel is ABI $(abi), expected 59"
[ -f "$BASE_OUT" ] || fail "gate base $BASE_OUT missing"

MAINBR=$(git -C "$R" rev-parse --abbrev-ref HEAD)
PRE=$(git -C "$R" rev-parse HEAD)
git -C "$WT" merge --no-edit "$MAINBR" >/dev/null 2>&1 || { git -C "$WT" merge --abort >/dev/null 2>&1; fail "merging $MAINBR ($PRE) into $BR conflicts"; }
say "merged $MAINBR ($PRE) into $BR: $(git -C "$WT" log --oneline -1 | cut -c1-90)"
( cd "$WT/src/wnn/ram/strategies/accelerator" && PYO3_PYTHON=$VENV/bin/python nice -n 5 cargo test -p ram_controller --lib --no-default-features ) > "$G/c60_cargo_test.log" 2>&1 \
	|| fail "cargo test -p ram_controller failed on the merged $BR (see $G/c60_cargo_test.log)"
say "cargo test ram_controller: $(grep 'test result' "$G/c60_cargo_test.log" | tail -1)"

# --- determinism gate: OLD = main tree on the installed ABI 59, NEW1/NEW2 = merged branch on 60 ---
rm -rf "$KEEP59" && mkdir -p "$KEEP59" && cp -a "$SITE"/ram_controller "$SITE"/ram_controller-*.dist-info "$KEEP59"/ || fail "cannot keep a copy of the ABI-59 wheel"
GATE="$WT/scripts/ctrl_determinism_gate.sh"
rm -rf "$G/c60_abi59" "$G/c60_abi60a" "$G/c60_abi60b" "$G/c60_smoke"
bash "$GATE" fly "$G/c60_abi59" "$R/src" "$BASE_OUT" >> "$G/c60_chain_gate.log" 2>&1 || fail "gate leg ABI 59 failed (see $G/c60_chain_gate.log)"
say "gate leg ABI 59 done"
( cd "$R" && unset CONDA_PREFIX && source wnn/bin/activate && cd "$WT/src/wnn/ram/strategies/accelerator" && maturin develop --release -m controller/Cargo.toml ) >> "$G/c60_chain_gate.log" 2>&1 \
	|| { restore59; fail "maturin develop (ABI 60) failed"; }
[ "$(abi)" = 60 ] || { restore59; fail "ABI 60 did not install (got $(abi))"; }
say "ABI 60 installed"
for r in c60_abi60a c60_abi60b; do
	bash "$GATE" fly "$G/$r" "$WT/src" "$BASE_OUT" >> "$G/c60_chain_gate.log" 2>&1 || { restore59; fail "gate leg $r failed"; }
done
if GATE_SRC=$WT/src bash "$GATE" compare "$G/c60_abi60a" "$G/c60_abi60b" "$G/c60_abi59" >> "$G/c60_chain_gate.log" 2>&1; then
	say "determinism gate PASS (flag-off ABI 60 x2 vs ABI 59, Y+F+L base)"
else
	restore59; fail "determinism gate FAIL — ABI 59 restored, nothing merged (see $G/c60_chain_gate.log)"
fi

# --- Python suites on the installed 60 ---
( cd "$WT/tests" && PYTHONPATH=$WT/src $VENV/bin/python -m pytest -q -p no:cacheprovider test_ctrl60_mid_window.py test_ctrl58_refine_fitness.py ) > "$G/c60_chain_pytest.log" 2>&1 \
	|| { restore59; fail "pytest (ctrl60 mid / ctrl58 refine) failed on ABI 60 — ABI 59 restored (see $G/c60_chain_pytest.log)"; }
( cd "$R" && PYTHONPATH=$WT/src $VENV/bin/python "$WT/tests/ctrl60_mid_scorer_parity.py" ) >> "$G/c60_chain_pytest.log" 2>&1 \
	|| { restore59; fail "ctrl60_mid_scorer_parity failed on ABI 60 — ABI 59 restored"; }
say "python: $(grep -E 'passed' "$G/c60_chain_pytest.log" | tail -1); mid scorer parity PASS"

# --- merge + push ---
git -C "$R" merge --ff-only "$BR" >/dev/null 2>&1 || git -C "$R" merge --no-edit "$BR" >/dev/null 2>&1 \
	|| { git -C "$R" merge --abort >/dev/null 2>&1; restore59; fail "merging $BR into $MAINBR failed — ABI 59 restored"; }
say "merged $BR into $MAINBR: $(git -C "$R" log --oneline -1 | cut -c1-90)"
PYTHONPATH=$R/src $VENV/bin/python -c "import ram_controller as c, wnn.control._accel as a; assert c.ABI_VERSION == 60 == a.EXPECTED_ABI" \
	|| fail "main tree facade / wheel are not both ABI 60 after the merge"
git -C "$R" push -q origin "$MAINBR" "$BR" >/dev/null 2>&1 && say "pushed $MAINBR and $BR" || say "WARNING: push failed (merge is local)"

# --- refine smoke: the gate-size Y+F+L run + the A2 flags must run clean and arm the mid column ---
mkdir -p "$G/c60_smoke"
sargv=$(PYTHONPATH=$R/src $VENV/bin/python "$R/scripts/ctrl_determinism_gate.py" argv "$BASE_OUT" "$G/c60_smoke") || fail "smoke argv build failed"
eval "set -- $sargv $EXTRA"
( cd "$R" && PYTHONPATH=$R/src $VENV/bin/python -u -m wnn.control.phased_ga "$@" ) > "$G/c60_smoke/run.out" 2>&1 \
	|| fail "refine smoke failed (see $G/c60_smoke/run.out) — ABI 60 stays installed, A2 NOT launched"
grep -aqiE "refine" "$G/c60_smoke/run.out" || fail "refine smoke ran but never mentions refine — A2 NOT launched"
grep -aqiE "mid" "$G/c60_smoke/run.out" || fail "refine smoke ran but shows no mid column — A2 NOT launched"
say "refine smoke OK ($G/c60_smoke/run.out)"

say "launching ab_a2.sh (A2 s1-s5, ABI 60)"
cd "$R" && exec bash "$L/ab_a2.sh"
