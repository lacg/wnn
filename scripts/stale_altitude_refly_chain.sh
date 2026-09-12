#!/usr/bin/env bash
# STALE-ALTITUDE RE-FLY — the paper rows of the altitude regimen, re-flown on the FIXED
# wheel (ABI 28, commit dc439d77). (12/09/2026, Luiz's order.)
#
# WHAT WAS WRONG. Every replay trainer (bptt_train_window / split_record /
# split_retrain_output) rebuilt its input frames through compute_features reading the
# controller's LAST LIVE vertical observation, so the three vertical bits
# (--obs-collective-cmd --obs-alt-err --obs-vz) were addressed at a STALE value on every
# replayed step. 148 banked runs / 594 h (14/08 -> 12/09) trained on that address —
# docs/controller_stale_altitude_rerun_inventory.md. The fix records vert_obs/horiz_obs per
# step and re-applies them before every compute_features (ReplayObs, fail-loud guard).
#
# THE CUT (Luiz, 12/09: "no need to run something that both were equally affected").
# Cohorts and sub-blocks whose only output was a RANKING between arms that all carried
# the bug (gatedwsweep, fitnessab, altweight, specialist1, stage1lambda, bits round 1 at
# n32, the round-2 bits x neurons x gamma grid) are NOT re-flown — their choice stands.
# What IS re-flown is every run whose ABSOLUTE number is a paper row or an arm paired
# against one:
#
#   step 1  anchors, seed 2:      SL_C_b32n64 _crn, SL_C_b28n256 _crn                   2 runs
#           (b24n256 s2 = the `_fix` run already banked; nothing else re-flies it)
#   step 2  b24n256 seeds 3,4,5   (the CRN bits-curve replication)                      3 runs
#   step 3  arms at b24n256, ROUND-MAJOR over seeds 2..5 so every arm has a paired
#           read after one round: _ls2 _ls4 _ls8 (arm A), _leak090, _mut1tap;
#           _win2 at seed 2 only (it was n=1)                                          21 runs
#   step 4  b32n256 seeds 2..6    (= the translation-ON arms, see MAPPING)              5 runs
#   step 5  b32n64 s2 leak 0.80 / 0.90   (= the leak revisit, see MAPPING)              2 runs
#                                                                                      -------
#                                                                              33 runs ~150 h
#
# MAPPING — three consolidations, so ONE runner (sweep_ladder_gamma.sh) flies everything
# and no arg list is duplicated:
#   TAB_on_b32n256_*_s{seed}  ==  SL_C_b32n256_*_g10_s{seed}  (translation_ab_chain.sh's ON
#       arm is byte-for-byte the ladder recipe: same FEAT, --translation, gamma 1.0)
#   LKR_l0X0_b32n64_*_s31337002  ==  SL_C_b32n64_*_g10_s31337002_leak0X0  (ladder + --delta-leak)
#   *_hd (derived teacher hover)  ==  plain  (--teacher-hover now DEFAULTS to derived since the
#       D0 verdict; every re-fly here passes it EXPLICITLY, so the plain reruns ARE the
#       derived-hover rows and a separate _hd re-fly would be a duplicate pair)
#   SL_C_b24n256_*_s31337002 (rotation era)  ==  ..._crn  ==  ..._fix  (CRN is the only scorer)
#
# ARCHIVE, NEVER DISCARD. Before the first launch every void marker in the list above is
# MOVED to experiments/<cohort>_markers_void_abi27/ (git keeps them; readers glob
# `*_markers/*.json`, so the void rows drop out of every leaderboard and paired read
# automatically). The .out / winner / ckpt of a tag re-flown under the SAME name move to
# logs/controller/sweep_ladder_void_abi27/ — the ckpt move matters: a stale
# emergency_stage*.yaml.gz from the void run would otherwise be RESUMED by the watchdog
# retry in run_controller_arm. A marker whose provenance already says abi>=28 is a
# fixed-wheel run and is left in place (idempotent re-entry).
#
# CONTROLS are only ever fixed-wheel markers: seed 2 at b24n256 -> the `_fix` run,
# seeds 3..5 -> the step-2 reruns. The verdict asserts provenance abi>=28 on every control
# it uses and refuses otherwise.
#
# GO-GATE. Luiz's call is conditional on the `_fix` vs void pair showing a material shift.
# This script therefore refuses to launch unless SAR_GO=1 is in the environment — arming
# it is the explicit "go", not a side effect of writing it.
#
# Marker-gated, idempotent, fails closed (a missing marker after a run = a human is
# needed; the box is left idle). HOLD sentinel honoured via run_controller_arm.
set -u

cd "$(dirname "$0")/.." || exit 1
. scripts/controller_arm_lib.sh

LOG="${SAR_LOG:-/private/tmp/stale_altitude_refly.log}"
VP="/Volumes/20260401-WDBlack-SN850X-2TB/wnn/venv/bin/python"
LADDER="scripts/sweep_ladder_gamma.sh"
EXP="${SAR_EXP:-experiments}"; LOGS="${SAR_LOGS:-logs/controller}"   # overridable for dry tests only
MARK="$EXP/sweepladder_markers";       VOID="$EXP/sweepladder_markers_void_abi27"
TABMARK="$EXP/translationab_markers";  TABVOID="$EXP/translationab_markers_void_abi27"
LKRMARK="$EXP/leakrevisit_markers";    LKRVOID="$EXP/leakrevisit_markers_void_abi27"
OUTDIR="$LOGS/sweep_ladder";           OUTVOID="$LOGS/sweep_ladder_void_abi27"
AIRFRAME="cf21_brushless"; DIST="L4C"
FIX_TAG="SL_C_b24n256_${AIRFRAME}_${DIST}_g10_s31337002_fix"
ARM_SEEDS="${SAR_ARM_SEEDS:-31337002 31337003 31337004 31337005}"
REP_SEEDS="${SAR_REP_SEEDS:-31337002 31337003 31337004 31337005 31337006}"
DMAX0="0.1"                                   # label_scale_arm_chain.sh: delta_max = 0.1/s
MUT_RATE="$($VP -c 'print(1/24)')"            # mutstep_ab_chain.sh: one tap per neuron at b=24
HOVER="--teacher-hover derived"               # explicit on every run (see MAPPING)

log() { echo "[stale-refly] $(date -u +%FT%TZ) $*" | tee -a "$LOG"; }
controller_pids() { pgrep -f "MacOS/Python -u -m wnn.control.phased_ga" 2>/dev/null || true; }
ladder_pids() { pgrep -f "scripts/sweep_ladder_gamma.sh" 2>/dev/null || true; }
busy() { [ -n "$(controller_pids)" ] || [ -n "$(ladder_pids)" ]; }

wait_box_clear() {
	local beat=0
	while busy; do
		sleep 30; beat=$((beat + 1))
		[ $((beat % 30)) = 0 ] && log "waiting — box busy"
	done
}

sl_tag() { # $1 bits  $2 neurons  $3 seed  $4 suffix
	echo "SL_C_b$1n$2_${AIRFRAME}_${DIST}_g10_s$3$4"
}

# The b24n256 control for an arm at a seed: the `_fix` run for seed 2, the step-2 rerun otherwise.
ctrl_tag() { # $1 seed
	[ "$1" = "31337002" ] && echo "$FIX_TAG" || sl_tag 24 256 "$1" ""
}

marker_abi() { # $1 marker path -> abi int, 0 when provenance is null/missing
	$VP - "$1" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
p = d.get("provenance") or {}
print(int(p.get("abi") or 0))
PY
}

fixed_marker() { # $1 marker path -> 0 when it is a fixed-wheel (abi>=28) marker
	[ -f "$1" ] && [ "$(marker_abi "$1")" -ge 28 ]
}

# ---- ARCHIVE -------------------------------------------------------------------------
# archive_marker <markdir> <voiddir> <tag> [same-name-rerun]  — move a VOID marker aside.
archive_marker() {
	local markdir="$1" voiddir="$2" tag="$3" same="${4:-}"
	local m="${markdir}/${tag}.json"
	[ -f "$m" ] || return 0
	if fixed_marker "$m"; then
		log "keep   ${tag} — already a fixed-wheel marker (abi $(marker_abi "$m"))"
		return 0
	fi
	mkdir -p "$voiddir"
	mv "$m" "${voiddir}/${tag}.json"
	log "VOID   ${tag} -> ${voiddir}/"
	if [ -n "$same" ]; then
		mkdir -p "$OUTVOID"
		local f
		for f in "${OUTDIR}/${tag}.out" "${OUTDIR}/${tag}_winner.yaml.gz" "${OUTDIR}/${tag}_winner.yaml.gz.fpga.json"; do
			[ -f "$f" ] && mv "$f" "${OUTVOID}/$(basename "$f")"
		done
		[ -d "${OUTDIR}/ckpt/${tag}" ] && { mkdir -p "${OUTVOID}/ckpt"; mv "${OUTDIR}/ckpt/${tag}" "${OUTVOID}/ckpt/${tag}"; }
	fi
}

write_void_readme() { # $1 voiddir  $2 one-line mapping note
	local readme="$1/README.md"
	[ -f "$readme" ] && return 0
	mkdir -p "$1"
	cat > "$readme" <<EOF
# VOID — flown with the stale-altitude-features bug (ABI <= 27)

Every marker here trained its vertical bits (--obs-collective-cmd --obs-alt-err --obs-vz)
at a STALE address: the replay trainers rebuilt frames from the controller's last live
vertical observation instead of the recorded per-step value. Fixed in commit dc439d77
(ram_controller ABI 28, wheel_sha256 f8aafb70e1dcd641); inventory in
docs/controller_stale_altitude_rerun_inventory.md.

Moved here by scripts/stale_altitude_refly_chain.sh on $(date -u +%FT%TZ). The fixed-wheel
re-fly carries a non-null \`provenance\` field (abi >= 28). Readers glob \`*_markers/*.json\`,
so nothing in this directory enters a leaderboard or a paired read.

$2
EOF
}

archive_all() {
	log "---------- ARCHIVE: moving void markers aside (nothing is deleted) ----------"
	write_void_readme "$VOID" "Same-name re-flies live in ../sweepladder_markers. Not re-flown, superseded by an identical recipe: SL_C_b24n256_*_s31337002 (rotation era) and ..._s31337002_crn == ..._s31337002_fix; *_hd == the plain re-fly (--teacher-hover derived is now the default and is passed explicitly)."
	write_void_readme "$TABVOID" "Not re-flown under these names: TAB_on_b32n256_*_s{seed} == SL_C_b32n256_*_g10_s{seed} (translation_ab_chain.sh's ON arm is the ladder recipe verbatim); the re-flies live in ../sweepladder_markers. TAB_off_* was never affected (no vertical features) and stays in ../translationab_markers."
	write_void_readme "$LKRVOID" "Not re-flown under these names: LKR_l0X0_b32n64_*_s31337002 == SL_C_b32n64_*_g10_s31337002_leak0X0 (ladder + --delta-leak); the re-flies live in ../sweepladder_markers."
	# superseded-by-identical-recipe (no same-name rerun)
	archive_marker "$MARK" "$VOID" "$(sl_tag 24 256 31337002 "")"
	archive_marker "$MARK" "$VOID" "$(sl_tag 24 256 31337002 _crn)"
	local s
	for s in $ARM_SEEDS; do archive_marker "$MARK" "$VOID" "$(sl_tag 24 256 "$s" _hd)"; done
	for s in $REP_SEEDS; do archive_marker "$TABMARK" "$TABVOID" "TAB_on_b32n256_${AIRFRAME}_${DIST}_s${s}"; done
	archive_marker "$LKRMARK" "$LKRVOID" "LKR_l080_b32n64_${AIRFRAME}_${DIST}_g10_s31337002"
	archive_marker "$LKRMARK" "$LKRVOID" "LKR_l090_b32n64_${AIRFRAME}_${DIST}_g10_s31337002"
	# same-name re-flies (marker + .out + winner + ckpt move together)
	archive_marker "$MARK" "$VOID" "$(sl_tag 32 64 31337002 _crn)" same
	archive_marker "$MARK" "$VOID" "$(sl_tag 28 256 31337002 _crn)" same
	for s in 31337003 31337004 31337005; do archive_marker "$MARK" "$VOID" "$(sl_tag 24 256 "$s" "")" same; done
	for s in $ARM_SEEDS; do
		local suf
		for suf in _ls2 _ls4 _ls8 _leak090 _mut1tap; do archive_marker "$MARK" "$VOID" "$(sl_tag 24 256 "$s" "$suf")" same; done
	done
	archive_marker "$MARK" "$VOID" "$(sl_tag 24 256 31337002 _win2)" same
	for s in $REP_SEEDS; do archive_marker "$MARK" "$VOID" "$(sl_tag 32 256 "$s" "")" same; done
	log "archive done: $(ls "$VOID" 2>/dev/null | grep -c json) sweepladder, $(ls "$TABVOID" 2>/dev/null | grep -c json) translationab, $(ls "$LKRVOID" 2>/dev/null | grep -c json) leakrevisit markers set aside"
}

# ---- ONE RUN -------------------------------------------------------------------------
# run_point <bits> <neurons> <seed> <suffix> <label> <extra-args> <extra-marker-json> <window_k>
run_point() {
	local b="$1" n="$2" seed="$3" suf="$4" label="$5" extra="$6" mj="$7" wk="$8"
	local tag; tag="$(sl_tag "$b" "$n" "$seed" "$suf")"
	local m="${MARK}/${tag}.json"
	if fixed_marker "$m"; then log "SKIP   ${tag} — banked on the fixed wheel"; return 0; fi
	if [ -f "$m" ]; then log "ABORT — ${tag} has a VOID marker in ${MARK} (archive step missed it). Box left idle."; exit 1; fi
	wait_box_clear
	log "===== START ${tag} (${label}; ${extra:-no extra args}) ====="
	SL_SKIP_PHASE1=1 SL_SWEEP_LABEL="$label" SL_FORCE_PHASE2_GAMMA="1.0" \
		SL_WIDTHS="$b" SL_NEURONS="$n" SL_SEED="$seed" SL_TAG_SUFFIX="$suf" SL_WINDOW_K="$wk" \
		SL_EXTRA_ARGS="${HOVER} ${extra}" \
		SL_EXTRA_MARKER_JSON=",\"refly\":\"stale-altitude-fix\",\"teacher_hover\":\"derived\"${mj}" \
		bash "$LADDER"
	log "ladder exited rc=$? for ${tag}"
	while [ -n "$(controller_pids)" ]; do sleep 30; done
	[ -f "$m" ] || { log "ABORT — marker ${tag}.json MISSING. A run needs a human. Box left idle."; exit 1; }
	fixed_marker "$m" || { log "ABORT — ${tag} banked WITHOUT abi>=28 provenance (wheel skew?). Box left idle."; exit 1; }
	log "banked: ${tag} (abi $(marker_abi "$m"))"
}

# ---- PREFLIGHT -----------------------------------------------------------------------
preflight() {
	[ "${SAR_GO:-}" = "1" ] || { log "REFUSED — SAR_GO=1 not set. Luiz's go is conditional on the _fix pair; arm explicitly."; exit 1; }
	busy && { log "ABORT — the box is NOT idle. Never launch this chain beside another."; exit 1; }
	local help; help=$(PYTHONPATH=src/wnn $VP -m wnn.control.phased_ga --help 2>/dev/null)
	local flag
	for flag in --teacher-hover --delta-label-scale --delta-max --delta-leak --conn-mutation-rate --input-window-k --conn-policy; do
		echo "$help" | grep -q -- "$flag" || { log "ABORT — the Python tree has no ${flag} (source/wheel skew)."; exit 1; }
	done
	PYTHONPATH=src/wnn $VP -c "import wnn.control._accel as a; assert a.EXPECTED_ABI >= 28, a.EXPECTED_ABI" 2>/dev/null \
		|| { log "ABORT — installed wheel/facade predate ABI 28 (the fix)."; exit 1; }
	fixed_marker "${MARK}/${FIX_TAG}.json" \
		|| { log "ABORT — ${FIX_TAG} is not banked with abi>=28. It is the seed-2 b24n256 anchor; nothing pairs without it."; exit 1; }
	[ -e experiments/HOLD_CONTROLLER ] && log "NOTE — HOLD sentinel present; the first run will wait until it is removed."
	log "preflight OK — SAR_GO=1, box idle, flags present, ABI>=28, ${FIX_TAG} banked."
}

# ---- VERDICT -------------------------------------------------------------------------
assert_fixed_controls() {
	local s missing=""
	for s in $ARM_SEEDS; do fixed_marker "${MARK}/$(ctrl_tag "$s").json" || missing="${missing} $(ctrl_tag "$s")"; done
	[ -z "$missing" ] || { log "VERDICT REFUSED — controls not fixed-wheel:${missing}"; return 1; }
}

void_vs_fixed_table() {
	log "---------- HOW BAD: void (abi<=27) vs fixed (abi 28) headline, same recipe ----------"
	$VP - "$MARK" "$FIX_TAG" "$VOID" "$TABVOID" "$LKRVOID" <<'PY'
import glob, json, math, os, re, sys
mark, fix, voids = sys.argv[1], sys.argv[2], sys.argv[3:]
K = math.log(0.5) / math.log(0.70)
def parse(p):
	try: d = json.load(open(p))
	except Exception: return None
	h = d.get("headline_holdout", "")
	g = lambda k: (re.search(k + r"=([0-9.]+)", h) or [None, None])[1]
	s, e, st, alt = g("stable"), g("err"), g("steady"), g("alt")
	if s is None or e is None: return None
	sf = max(float(s) / 100.0, 1e-6)
	hd = 0.5556 * (float(e) / 8.0) + 0.4444 * min(K * -math.log2(sf), 20.0)
	return dict(stable=float(s), err=float(e), steady=float(st or "nan"), alt=float(alt or "nan"), hd=hd)
def fixed_tag(tag):
	"""The fixed-wheel tag that carries the SAME recipe as a void tag (see MAPPING in the header)."""
	s2 = "s31337002"
	if tag.startswith("SL_C_b24n256_") and tag.endswith((s2, s2 + "_crn", s2 + "_hd")):
		return fix
	if tag.endswith("_hd"):
		return tag[:-3]
	m = re.match(r"TAB_on_(b32n256_.*)_s(\d+)$", tag)
	if m: return "SL_C_%s_g10_s%s" % (m.group(1), m.group(2))
	m = re.match(r"LKR_l(\d+)_(b32n64_.*_g10_s\d+)$", tag)
	if m: return "SL_C_%s_leak%s" % (m.group(2), m.group(1))
	return tag
rows = []
for void in voids:
	for vp in sorted(glob.glob(os.path.join(void, "*.json"))):
		tag = os.path.basename(vp)[:-5]
		ft = fixed_tag(tag)
		fp = os.path.join(mark, ft + ".json")
		v, f = parse(vp), parse(fp) if os.path.exists(fp) else None
		if v and f: rows.append((tag, ft, v, f))
if not rows:
	print("  (no fixed/void pair banked yet)"); sys.exit(0)
print("  %-56s %9s %9s %9s %9s %9s" % ("void tag  (-> fixed tag when renamed)", "d_stable", "d_err", "d_steady", "d_alt", "d_hd"))
for tag, ft, v, f in rows:
	name = tag if ft == tag else "%s -> %s" % (tag, ft)
	print("  %-56s %+8.1f%% %+8.2f° %+8.2f° %+8.3fm %+9.4f" % (name, f["stable"] - v["stable"], f["err"] - v["err"], f["steady"] - v["steady"], f["alt"] - v["alt"], f["hd"] - v["hd"]))
n = len(rows)
for k in ("err", "alt", "hd"):
	ds = [f[k] - v[k] for _, _, v, f in rows]
	m = sum(ds) / n
	sd = (sum((d - m) ** 2 for d in ds) / (n - 1)) ** 0.5 if n > 1 else float("nan")
	ci = 1.96 * sd / n ** 0.5 if n > 1 else float("nan")
	print("  mean d_%s = %+.4f  (n=%d, 95%% CI ±%.4f)  — fixed minus void; lower hd = better" % (k, m, n, ci))
shared = len(rows) - len({ft for _, ft, _, _ in rows})
if shared:
	print("  CAVEAT: %d row(s) share a fixed tag — they are ONE measurement, so n is inflated by that much." % shared)
	print("          At seed 2 the one-flag pair is _hd -> _fix (both derived hover); _crn also differs by hover,")
	print("          plain (rotation era) by hover AND scorer.")
PY
}

verdict() {
	$VP scripts/gate_distance_leaderboard.py > docs/controller_gate_distance_leaderboard.md 2>/dev/null
	void_vs_fixed_table
	assert_fixed_controls || return 1
	log "---------- ARMS: mean paired delta + 95% CI vs the FIXED b24n256 controls (NOT a win tally) ----------"
	local suf
	for suf in _ls2 _ls4 _ls8 _leak090 _mut1tap; do
		log "arm ${suf}:"
		PYTHONPATH=src/wnn $VP scripts/paired_power.py \
			--primary err --arm "$suf" \
			--base "SL_C_b24n256_${AIRFRAME}_${DIST}_g10_s{seed}" \
			$(for s in $ARM_SEEDS; do printf -- '--seed %s ' "$s"; done) \
			--control-override "31337002=${FIX_TAG}" 2>&1 | tee -a "$LOG"
	done
	log "_win2 and the b32n64 leaks are n=1 — direction only; read them off the leaderboard."
	log "READ: four columns per stage, same seed, lower hd = better. n=4 bounds an arm at ~0.6° err."
}

# ---- DRY RUN -------------------------------------------------------------------------
# SAR_DRY=1 SAR_EXP=<copy of experiments> SAR_LOGS=<copy of logs/controller>: archive on the
# COPY, print the plan and the how-bad table, launch nothing. Refuses the real tree.
plan_line() { # $1 tag
	local m="${MARK}/$1.json"
	if fixed_marker "$m"; then echo "  banked(fixed)  $1"; elif [ -f "$m" ]; then echo "  VOID-IN-PLACE  $1  <- archive missed it"; else echo "  to fly         $1"; fi
}
dry_plan() {
	log "---------- PLAN (33 runs) ----------"
	plan_line "$(sl_tag 32 64 31337002 _crn)"; plan_line "$(sl_tag 28 256 31337002 _crn)"
	local s S
	for s in 31337003 31337004 31337005; do plan_line "$(sl_tag 24 256 "$s" "")"; done
	for s in $ARM_SEEDS; do
		for S in 2 4 8; do plan_line "$(sl_tag 24 256 "$s" "_ls${S}")"; done
		plan_line "$(sl_tag 24 256 "$s" _leak090)"; plan_line "$(sl_tag 24 256 "$s" _mut1tap)"
		[ "$s" = "31337002" ] && plan_line "$(sl_tag 24 256 "$s" _win2)"
	done
	for s in $REP_SEEDS; do plan_line "$(sl_tag 32 256 "$s" "")"; done
	plan_line "$(sl_tag 32 64 31337002 _leak080)"; plan_line "$(sl_tag 32 64 31337002 _leak090)"
}
if [ "${SAR_DRY:-}" = "1" ]; then
	[ "$EXP" != "experiments" ] && [ "$LOGS" != "logs/controller" ] \
		|| { log "DRY REFUSED — point SAR_EXP/SAR_LOGS at a COPY; a dry run never touches the real tree."; exit 1; }
	log "########## DRY RUN on $EXP / $LOGS — nothing launches ##########"
	archive_all
	dry_plan
	void_vs_fixed_table
	exit 0
fi

# ---- CHAIN ---------------------------------------------------------------------------
log "########## ARMED — stale-altitude re-fly, paper rows only (33 runs) ##########"
preflight
archive_all
dry_plan

log "===== STEP 1 anchors (seed 31337002): b32n64 _crn, b28n256 _crn ====="
run_point 32 64  31337002 _crn crn-refly "" ",\"arm\":\"anchor\"" 1
run_point 28 256 31337002 _crn crn-refly "" ",\"arm\":\"anchor\"" 1

log "===== STEP 2 b24n256 seeds 3,4,5 (CRN bits-curve replication; the arms' controls) ====="
for s in 31337003 31337004 31337005; do
	run_point 24 256 "$s" "" crn-bits-curve "" ",\"arm\":\"anchor\"" 1
done

log "===== STEP 3 arms at b24n256, round-major over seeds [${ARM_SEEDS}] ====="
for s in $ARM_SEEDS; do
	for S in 2 4 8; do
		DMAX="$($VP -c "print(${DMAX0}/${S})")"
		run_point 24 256 "$s" "_ls${S}" label-scale "--delta-label-scale ${S} --delta-max ${DMAX}" \
			",\"arm_ls\":${S},\"delta_label_scale\":${S},\"delta_max\":${DMAX},\"control_tag\":\"$(ctrl_tag "$s")\"" 1
	done
	run_point 24 256 "$s" _leak090 leak090 "--delta-leak 0.90" \
		",\"arm\":\"leak090\",\"delta_leak\":0.90,\"control_tag\":\"$(ctrl_tag "$s")\"" 1
	run_point 24 256 "$s" _mut1tap mutstep-ab "--conn-mutation-rate ${MUT_RATE}" \
		",\"arm_b\":\"mut1tap\",\"conn_mutation_rate\":${MUT_RATE},\"control_tag\":\"$(ctrl_tag "$s")\"" 1
	if [ "$s" = "31337002" ]; then
		run_point 24 256 "$s" _win2 win2 \
			"--conn-policy framed1 --output-full-window --input-window-k 2 --frame-stride 10 --conn-mutation-scope window" \
			",\"arm\":\"win2\",\"conn_policy\":\"framed1\",\"output_full_window\":true,\"frame_stride\":10,\"conn_mutation_scope\":\"window\",\"recency_weights\":\"2^slot\",\"control_tag\":\"$(ctrl_tag "$s")\"" 2
	fi
done

log "===== STEP 4 b32n256 seeds [${REP_SEEDS}] (= the translation-ON arms) ====="
for s in $REP_SEEDS; do
	run_point 32 256 "$s" "" b32n256-replication "" ",\"arm\":\"anchor\",\"was\":\"TAB_on\"" 1
done

log "===== STEP 5 b32n64 seed 31337002 leak 0.80 / 0.90 (= the leak revisit) ====="
for leak in 0.80 0.90; do
	run_point 32 64 31337002 "_leak$(echo "$leak" | tr -d '.')" leak-revisit "--delta-leak ${leak}" \
		",\"arm\":\"leak$(echo "$leak" | tr -d '.')\",\"delta_leak\":${leak},\"was\":\"LKR\",\"control_tag\":\"$(sl_tag 32 64 31337002 _crn)\"" 1
done

verdict
log "########## STALE-ALTITUDE RE-FLY COMPLETE — 33/33 markers on the fixed wheel ##########"
