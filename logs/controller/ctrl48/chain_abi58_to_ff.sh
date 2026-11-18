#!/usr/bin/env bash
# CTRL-48 overnight chain (Luiz 17/11 ~23:10 EST): DMY DONE → CTRL-56 watcher finished → FF verdict GO →
# ABI 58 merged-on-main + cargo tests → determinism gate (flag-off ABI 58 vs ABI 57) → merge + push →
# prereg launch addendum → FF block (ab_yawff.sh, detached). Every step is a "FF CHAIN:" line in runs.log;
# every failure REFUSES (no install / restores the ABI-57 wheel) and says why.
set -u
R=/Users/lacg/wnn
WT=/Users/lacg/wnn-abi58
BR=ctrl48-abi58
L=$R/logs/controller/ctrl48
RUNS=$L/runs.log
VENV=/Volumes/20260401-WDBlack-SN850X-2TB/wnn/venv
SITE=$VENV/lib/python3.13/site-packages
KEEP57=/Users/lacg/wnn-abi57-wheel-keep
G=$R/logs/controller/detgate
VERDICT=$L/l2yp_investigation/ff_precheck_verdict.txt
CTRL56_PID=${CTRL56_PID:-71647}
CTRL56_LOG=$R/logs/controller/ctrl56_gate/merge.log
PREREG=docs/ctrl48_yawff_prereg.md
say() { echo "$(date -u +%FT%TZ) FF CHAIN: $*" >> "$RUNS"; echo "[ff-chain] $(TZ=America/New_York date '+%d/%m %H:%M %Z') $*"; }
fail() { say "REFUSED: $*"; exit 1; }
abi() { "$VENV/bin/python" -c "import ram_controller as c; print(c.ABI_VERSION)" 2>/dev/null; }
restore57() {
	if [ -d "$KEEP57/ram_controller" ]; then
		rm -rf "$SITE/ram_controller" "$SITE"/ram_controller-*.dist-info
		cp -a "$KEEP57"/ram_controller "$KEEP57"/ram_controller-*.dist-info "$SITE"/
		say "restored the ABI-57 wheel (now ABI $(abi))"
	fi
}

say "armed (pid $$): waiting for DMY DONE, then the CTRL-56 watcher (pid $CTRL56_PID)"
until grep -q "DMY DONE" "$RUNS"; do sleep 120; done
say "DMY DONE seen"
while kill -0 "$CTRL56_PID" 2>/dev/null; do sleep 30; done
say "CTRL-56 watcher finished: $(grep '\[ctrl56-merge\]' "$CTRL56_LOG" | tail -1 | cut -c1-200)"
[ -f "$R/.git/MERGE_HEAD" ] && fail "the main tree is mid-merge (CTRL-56 left a conflict) — not touching it"
pgrep -f "wnn.control.phased_g[a]" >/dev/null && fail "a phased_ga is running"
[ "$(abi)" = 57 ] || fail "installed wheel is ABI $(abi), expected 57"

# --- the FF pre-check verdict (line 1 GO / NO-GO; optional BASE: / EXPECT / SPEC-DELTA: sections) ---
waited=0
until [ -f "$VERDICT" ]; do
	[ $waited -ge 10800 ] && fail "no verdict file after 3 h ($VERDICT)"
	sleep 120; waited=$((waited + 120))
done
line1=$(head -1 "$VERDICT" | tr -d '[:space:]')
say "verdict line 1: '$line1'"
[ "$line1" = "GO" ] || fail "verdict is not GO ('$line1') — ABI 58 NOT installed, no block"
if awk '/^SPEC-DELTA:/{f=1;sub(/^SPEC-DELTA:[ \t]*/,"");if($0!=""&&tolower($0)!~/^none/)print;next} f&&/^[A-Z-]+:/{f=0} f&&NF{print}' "$VERDICT" | grep -qv -i '^none'; then
	fail "the verdict carries a SPEC-DELTA — it must be applied to the build before the block (not automatic)"
fi
if grep -qiE '^BASE:.*(delta-max-yaw|\bY\b)' "$VERDICT"; then BASE=Y; else BASE=M; fi
say "FF block base = $BASE (verdict BASE line: '$(grep -i '^BASE:' "$VERDICT" | head -1)')"

# --- ABI 58 on top of whatever main now holds ---
MAINBR=$(git -C "$R" rev-parse --abbrev-ref HEAD)
PRE=$(git -C "$R" rev-parse HEAD)
git -C "$WT" merge --no-edit "$MAINBR" >/dev/null 2>&1 || { git -C "$WT" merge --abort >/dev/null 2>&1; fail "merging $MAINBR ($PRE) into $BR conflicts"; }
say "merged $MAINBR ($PRE) into $BR: $(git -C "$WT" log --oneline -1)"
( cd "$WT/src/wnn/ram/strategies/accelerator" && PYO3_PYTHON=$VENV/bin/python nice -n 5 cargo test -p ram_controller --lib --no-default-features ) > "$WT/chain_cargo_test.log" 2>&1 \
	|| fail "cargo test failed on the merged $BR (see $WT/chain_cargo_test.log)"
say "cargo test: $(grep 'test result' "$WT/chain_cargo_test.log" | tail -1)"

# --- determinism gate: OLD = main tree (ABI 57 installed), NEW1/NEW2 = merged branch on ABI 58 ---
rm -rf "$KEEP57" && mkdir -p "$KEEP57" && cp -a "$SITE"/ram_controller "$SITE"/ram_controller-*.dist-info "$KEEP57"/ || fail "cannot keep a copy of the ABI-57 wheel"
GATE="$WT/scripts/ctrl_determinism_gate.sh"
rm -rf "$G/ff_abi57" "$G/ff_abi58a" "$G/ff_abi58b"
bash "$GATE" fly "$G/ff_abi57" "$R/src" >> "$G/ff_chain_gate.log" 2>&1 || fail "gate leg ABI 57 failed (see $G/ff_chain_gate.log)"
say "gate leg ABI 57 done"
( cd "$R" && unset CONDA_PREFIX && source wnn/bin/activate && cd "$WT/src/wnn/ram/strategies/accelerator" && maturin develop --release -m controller/Cargo.toml ) >> "$G/ff_chain_gate.log" 2>&1 \
	|| { restore57; fail "maturin develop (ABI 58) failed"; }
[ "$(abi)" = 58 ] || { restore57; fail "ABI 58 did not install (got $(abi))"; }
say "ABI 58 installed"
for r in ff_abi58a ff_abi58b; do
	bash "$GATE" fly "$G/$r" "$WT/src" >> "$G/ff_chain_gate.log" 2>&1 || { restore57; fail "gate leg $r failed"; }
done
if GATE_SRC=$WT/src bash "$GATE" compare "$G/ff_abi58a" "$G/ff_abi58b" "$G/ff_abi57" >> "$G/ff_chain_gate.log" 2>&1; then
	say "determinism gate PASS (flag-off ABI 58 x2 vs ABI 57)"
else
	restore57; fail "determinism gate FAIL — ABI 57 restored, nothing merged (see $G/ff_chain_gate.log)"
fi

# --- flag-ON smoke: the gate's small run + --yaw-trim-ff (base flags of the block), must exit 0 ---
SMK=$G/ff_smoke_flag_on
rm -rf "$SMK" && mkdir -p "$SMK"
SARGV=$(PYTHONPATH=$WT/src $VENV/bin/python "$WT/scripts/ctrl_determinism_gate.py" argv "$L/C-S2-d_wposx2_a55_rf_c200_dc_biasin_ms01_P_both_L1_yfeM_full_s1.out" "$SMK") \
	|| { restore57; fail "flag-on smoke: argv build failed"; }
[ "$BASE" = Y ] && SARGV="$SARGV --delta-max-yaw 0.5"
SARGV="$SARGV --yaw-trim-ff"
echo "$SARGV" > "$SMK/argv"
( eval "set -- $SARGV"; cd "$R" && PYTHONPATH=$WT/src $VENV/bin/python -u -m wnn.control.phased_ga "$@" ) > "$SMK/run.out" 2>&1 \
	|| { restore57; fail "flag-on smoke run failed (see $SMK/run.out) — ABI 57 restored"; }
grep -q "yawref=[a-z_+]*ff" "$SMK/run.out" || { restore57; fail "flag-on smoke: provenance lacks yawref=…ff"; }
say "flag-on smoke PASS (rc 0, $(grep -oE 'yawref=[a-z_+]+' "$SMK/run.out" | head -1), $(grep -c 'Gen [0-9]*/' "$SMK/run.out") gen lines)"

# --- merge + push, then the prereg launch addendum, then the block ---
git -C "$R" merge --ff-only "$BR" >/dev/null 2>&1 || git -C "$R" merge --no-edit "$BR" >/dev/null 2>&1 \
	|| { git -C "$R" merge --abort >/dev/null 2>&1; restore57; fail "merging $BR into $MAINBR failed — ABI 57 restored"; }
say "merged $BR into $MAINBR: $(git -C "$R" log --oneline -1)"
PYTHONPATH=$R/src $VENV/bin/python -c "import ram_controller as c, wnn.control._accel as a; assert c.ABI_VERSION == 58 == a.EXPECTED_ABI" \
	|| fail "main tree facade / wheel are not both ABI 58 after the merge"
{
	echo
	echo "## Launch addendum (written by chain_abi58_to_ff.sh, $(TZ=America/New_York date '+%d/%m/%Y %H:%M %Z'), before the first FF START)"
	echo "- Base: **$BASE** (from the verdict file's BASE line; absent ⇒ M)."
	echo "- Determinism gate: PASS (flag-off ABI 58 ×2 vs ABI 57; logs/controller/detgate/ff_abi5*)."
	echo "- Expectation, frozen verbatim from $VERDICT:"
	echo '```'
	grep -iE '^(EXPECT|PREDICT)' -A6 "$VERDICT" || echo "(the verdict states no EXPECT/PREDICT section)"
	echo '```'
} >> "$R/$PREREG"
git -C "$R" add "$PREREG" && git -C "$R" commit -q -m "CTRL-48 FF block launch addendum (base $BASE, gate PASS) — chain_abi58_to_ff.sh

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Exc8QwHhxZqBnX4jPhZ3aZ" >/dev/null 2>&1 || say "WARNING: prereg addendum commit failed (block still launches)"
git -C "$R" push -q origin "$MAINBR" "$BR" >/dev/null 2>&1 && say "pushed $MAINBR and $BR" || say "WARNING: push failed (merge is local)"
pgrep -f "wnn.control.phased_g[a]" >/dev/null && fail "a phased_ga started meanwhile — not launching"
FF_BASE=$BASE nohup bash "$L/ab_yawff.sh" > "$L/ab_yawff.driver.log" 2>&1 &
say "FF block launched: ab_yawff.sh pid $! base $BASE"
