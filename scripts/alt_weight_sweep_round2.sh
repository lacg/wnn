#!/bin/bash
# ALT-WEIGHT SWEEP, ROUND 2 — a second seed on the survivors only.
#
# WHY THIS EXISTS. Round 1 flies each of the 9 arms ONCE (seed 31337002). That is the
# broad net, and n=1 is exactly the exposure that has been biting this programme: at
# b=14 and again at b=18 the selected genome landed far off its own block, because one
# draw decided it. Choosing the fitness weights we will then fly for ALL of stage A off
# a single seed would bake that lottery into every later run.
#
# So: round 1 screens, round 2 confirms. Only the top arms get a second seed
# (31337003 — the ladder's own pair, so the numbers stay comparable), and the winner is
# the LOWEST MEAN steady across its two seeds. Never best-of-N: taking the better of two
# draws is how you re-import the variance you just paid to remove.
#
# THE PRE-REGISTERED FILTER, applied before spending 3 h on an arm. An arm is eligible
# only if its held-out altitude error did NOT regress against its OWN base's alt=0.00
# control. That is the sweep's stated read — buy altitude WITHOUT selling attitude — and
# an arm that wins steady by abandoning altitude has not won, so it does not deserve a
# second seed. Ineligible arms are logged with their numbers, never silently dropped.
#
# NO DUPLICATION. The recipe, the weights and the arm() function all live in
# alt_weight_sweep_chain.sh. This script EXTRACTS and evals them rather than retyping —
# the same trick recalc_sweep_headlines.sh uses for the config block. Re-typing a 25-flag
# recipe is how two runs silently stop being comparable. Because the tag string embeds
# ${SEED}, overriding SEED regenerates the correct round-2 tags for free.
#
#   Usage:  scripts/alt_weight_sweep_round2.sh [--dry-run]
#     AW_ROUND2_K     how many survivors get a second seed (default 3)
#     AW_ROUND2_SEED  the second seed (default 31337003)
set -u

ROOT="/Users/lacg/wnn"
cd "$ROOT" || exit 1
CHAIN="$ROOT/scripts/alt_weight_sweep_chain.sh"
VP="/Volumes/20260401-WDBlack-SN850X-2TB/wnn/venv/bin/python"
MARKDIR="${AW_MARKDIR:-experiments/altweight_markers}"   # overridable so the ranking
                                                        # can be tested on synthetic
                                                        # markers before round 1 lands
LOG="/private/tmp/alt_weight_sweep.log"
K="${AW_ROUND2_K:-3}"
DRY=0
if [ "${1:-}" = "--dry-run" ]
then
	DRY=1
fi

# Round 1 must be complete: ranking a partial round would pick survivors from whichever
# arms happened to finish first, which is an ordering artifact, not a result.
HAVE=$(ls "$MARKDIR"/AW_*_s31337002.json 2>/dev/null | wc -l | tr -d ' ')
if [ "$HAVE" -lt 9 ]
then
	echo "round 1 is $HAVE/9 — refusing to rank a partial round (an arm that has not"
	echo "flown cannot be judged, and survivors would reflect finishing order)."
	exit 1
fi

# --- rank + filter -----------------------------------------------------------------
# Prints one line per arm: "<tag> <steady> <alt> <verdict>", best steady first.
TABLE=$("$VP" - "$MARKDIR" <<'PY'
import glob, json, os, re, sys

markdir = sys.argv[1]
rows = []
for path in sorted(glob.glob(os.path.join(markdir, "AW_*_s31337002.json"))):
	try:
		d = json.load(open(path))
	except Exception:
		continue
	head = d.get("headline_holdout") or ""
	steady = re.search(r"steady=([0-9.]+)", head)
	alt    = re.search(r"alt=([0-9.]+)", head)
	stable = re.search(r"stable=([0-9.]+)", head)
	err    = re.search(r"err=([0-9.]+)", head)
	if steady is None:
		continue
	tag = d.get("tag", os.path.basename(path)[:-5])
	base = "C10" if "_C10_" in tag else ("S16" if "_S16_" in tag else "REF")
	rows.append({
		"tag": tag, "base": base,
		"steady": float(steady.group(1)),
		"alt": float(alt.group(1)) if alt else float("nan"),
		"stable": float(stable.group(1)) if stable else float("nan"),
		"err": float(err.group(1)) if err else float("nan"),
		"is_control": "_alt000_" in tag,
	})

# Each base's alt=0.00 control sets the altitude bar for that base. REF has no control
# of its own (it is the OLD scheme) and is always eligible — it exists to be compared.
bar = {r["base"]: r["alt"] for r in rows if r["is_control"]}
for r in rows:
	if r["base"] == "REF" or r["is_control"]:
		r["verdict"] = "eligible"
	elif r["base"] not in bar:
		r["verdict"] = "eligible(no-control)"
	elif r["alt"] <= bar[r["base"]]:
		r["verdict"] = "eligible"
	else:
		r["verdict"] = "ALT-REGRESSED"

rows.sort(key=lambda r: r["steady"])
for r in rows:
	print("%s %.4f %.4f %s %.1f %.2f" % (
		r["tag"], r["steady"], r["alt"], r["verdict"], r["stable"], r["err"]))
PY
)

log()
{
	echo "[altsweep-r2] $(date -u +%FT%TZ) $*" | tee -a "$LOG"
}

log "########## ROUND 1 TABLE (ranked by held-out steady; alt bar = each base's alt000) ##########"
printf '%-42s %8s %8s %-20s %7s %7s\n' TAG STEADY ALT VERDICT STABLE ERR | tee -a "$LOG"
while IFS=' ' read -r tag steady alt verdict stable err
do
	printf '%-42s %8s %8s %-20s %7s %7s\n' "$tag" "$steady" "$alt" "$verdict" "$stable" "$err" | tee -a "$LOG"
done <<< "$TABLE"

SURVIVORS=$(echo "$TABLE" | awk '$4 ~ /^eligible/ {print $1}' | head -"$K")
if [ -z "$SURVIVORS" ]
then
	log "ABORT: no arm passed the altitude filter — every weighted arm regressed altitude"
	log "against its own control. That is itself the finding; do NOT fly a round 2."
	exit 1
fi
log "########## SURVIVORS (top $K eligible by steady): $(echo $SURVIVORS | tr '\n' ' ') ##########"

if [ "$DRY" = "1" ]
then
	log "--dry-run: nothing flown"
	exit 0
fi

# --- re-fly the survivors at the second seed ---------------------------------------
# Pull the config block, the helpers and the arm() definition out of round 1 rather than
# restating them. SEED is overridden AFTER the eval so the tags regenerate correctly.
eval "$(sed -n '/^ROOT=/,/^FEAT_STAGE1=/p' "$CHAIN")"
eval "$(sed -n '/^log()$/,/^}$/p' "$CHAIN")"
eval "$(sed -n '/^arm()$/,/^}$/p' "$CHAIN")"
SEED="${AW_ROUND2_SEED:-31337003}"

log "########## ARMED — ROUND 2, seed ${SEED}, $(echo $SURVIVORS | wc -w | tr -d ' ') arm(s) ##########"
while IFS= read -r tag
do
	[ -z "$tag" ] && continue
	# Recover this arm's exact weights by finding its invocation in round 1 and
	# re-evaluating it with the new SEED, so the recipe cannot drift between seeds.
	#
	# Match on the part of the tag BEFORE "_b" (e.g. AW_C10_alt020). The marker's tag is
	# fully expanded (AW_C10_alt020_b18n32_s31337002) while the script still holds
	# "AW_C10_alt020_b${BITS}n${NEURONS}_s${SEED}" — so matching the whole stem can never
	# hit. That prefix is unique per arm by construction.
	stem="${tag%%_b*}"
	matches=$(grep -c "^arm \"${stem}_b" "$CHAIN")
	if [ "$matches" != "1" ]
	then
		log "$tag: expected exactly 1 arm() invocation matching '${stem}_b' in $CHAIN, found $matches — SKIPPED"
		continue
	fi
	line=$(grep "^arm \"${stem}_b" "$CHAIN")
	log "round2: $tag  ->  $line"
	eval "$line"
done <<< "$SURVIVORS"

log "########## ROUND 2 COMPLETE — $(ls "$MARKDIR" | wc -l | tr -d ' ') markers total ##########"
