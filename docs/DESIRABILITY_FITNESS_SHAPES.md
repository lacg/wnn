# Desirability fitness — utility shape draft (FOR DISCUSSION, nothing implemented)

Drafted 26/08/2026 from Luiz's spec: one continuous formula whose LIMIT behavior
does what the gate did, whose weights are never inert, and which is
substrate-generic (same code, controller and IDS). Mathematical family:
multiplicative utility (Cobb-Douglas; Derringer–Suich desirability).

## The formula

Per metric: a utility `u_i(x) ∈ (eps, 1]`, 1 = ideal, →eps = unacceptable.

    fitness = Π u_i(x_i)^w_i          (higher = better)

Equivalently, and this is the form the code should use (matches the existing
"lower = better" combine contract, avoids float underflow, and is additive):

    score = -log2(fitness) = Σ w_i · h_i(x_i)     where h_i = -log2(u_i)

`h_i` reads as "how many HALF-LIVES of desirability metric i has lost" — the
whole score is a weighted count of half-lives lost. Lower = better. The product
becomes a sum, but in LOG-utility space, which is where the dynamic exchange
rate lives: a metric near unacceptable has enormous h, so trading it further is
ruinous, while metrics near ideal trade almost freely. The tumbling genome
(stable≈0) carries h_stable ≈ 20 half-lives and cannot be bought back by
smoothness — the gate's job, now emergent.

Floors: u clamped at eps = 2^-20 (h capped at 20). Keeps every ordering strict
and a gradient alive even among total failures (the old gen-0 "distance to
flying" regime falls out of the same formula).

## Controller shapes (anchors are MEASURED, sources inline)

Data: 1,830 held-out RESULT rows, gated weight sweep (the feasible regime —
anchors placed where discrimination among GOOD flyers matters). Medians:
stable .87, err 3.19°, steady 3.03°, jerk .031, mono 1.0, alt 0.52m.
PID reference on matched seed: 100% / 1.24° / 0.55°.

| metric  | shape                       | half-anchor (u=0.5)         | rationale                                        | sanity points                                    |
|---------|-----------------------------|------------------------------|--------------------------------------------------|--------------------------------------------------|
| stable  | u = s^1.943                 | s = 0.70                     | gate threshold RETAINED as the concern point     | u(.87)=.76 · u(.96)=.92 · u(.33)=.12 · u(.10)=.011 |
| err     | u = 2^(−e/8.0)              | e = 8.0°                     | gate threshold RETAINED; each 8° halves u        | u(3.19)=.76 · u(1.24 PID)=.90 · u(26.7)=.10       |
| steady  | u = 2^(−d/8.0)              | d = 8.0°                     | symmetric with err (no prior threshold existed)  | u(3.03)=.77 · u(0.55 PID)=.95 · u(17.6)=.22       |
| jerk    | u = 2^(−j/0.06)             | j = 0.06                     | ~2× flyer median; observed flyers all ≤ 0.039    | u(.031)=.70 · u(.039)=.64 — gentle by design      |
| mono    | u = 2^(−m/2.0)              | m = 2.0                      | counts; median flyer = 1                         | u(1)=.71 · u(5)=.18                               |
| alt     | u = 2^(−a/1.0)              | a = 1.0 m                    | flyer median 0.52m; worst 3.84m                  | u(.52)=.70 · u(3.84)=.07                          |

Weights (exponents): **start with S16noJM unchanged** — err .3125, stable .25,
steady .4375, jerk/mono/alt 0 — so the A/B isolates the AGGREGATION change from
any weight change. Weight questions (does jerk deserve >0 now that tumblers
can't exploit it?) come AFTER the A/B, not inside it.

Semantic shift to be aware of: the current column is err SQUARED (quadratic
penalty in sum space). Exponential utility is log-linear → the PRODUCT penalty
grows exponentially in degrees — harsher than quadratic at large errors,
gentler near zero. This is a change in emphasis, not a bug; it is the reason
case-3 (90%/100m) flips.

## IDS shapes (same code, different anchor table)

Anchors from banked results: production Wb F1 93.34 / FPR 8.37 (XGBoost
parity); UNSW-temp 16b-Wb 88.86 / 8.78.

| metric        | shape                | half-anchor        | rationale                                          |
|---------------|----------------------|--------------------|----------------------------------------------------|
| f1            | u = f1^3.106         | f1 = 0.80          | below 80% F1 is not a competitive IDS              |
| fpr           | u = 2^(−fpr/0.10)    | fpr = 0.10         | 10% FPR = operator pain threshold; u(.0837)=.56     |
| ce            | u = 2^(−ce/c½)       | c½ = cohort median | fit once per dataset family, then FROZEN           |
| accuracy      | u = acc^k, w = 0     | —                  | shape drafted, weight 0: banked "weighting accuracy HURTS" (IDSZ/IDSX) |
| recall_c ×K   | u_c = r_c^1.0, w/K each | —               | THE anti-QSR device: per-class recalls enter the product, so an aggregate win bought with 8/9 recall collapses multiplies eight ~0 utilities and DIES |
| benign_fpr    | u = 2^(−bfpr/0.10)   | 0.10               | multiclass benign column                            |

The per-class-recall row is the payoff of genericity: multiclass fitness =
macro terms × Π_c recall_c^(w_r/K). No IDS-specific aggregation code — just a
longer metric vector into the same combine.

## Where it lives

Fourth `combine_flat` mode in ram_core::fitness (`"desirability"`), taking per
column: (shape enum, half_anchor, weight). Controller consumes it in place of
gate+violation+zscore; IDS through the existing fitness_aggregation plumbing.
`gated_combine_flat` remains for comparison arms; Pareto (ON HOLD, Luiz) would
be an audit/report layer, not the ranking.

## Pre-registered A/B (required before adoption — this changes the trajectory)

Two arms × matched seeds, identical everything except aggregation:
  A = current (gate 0.70/8.0 + Deb + zscore + S16noJM)
  B = desirability (shapes above + S16noJM exponents)
Primary read: paired per-seed held-out stable% and err°; steady always quoted.
Also log per-generation: fraction of pool at u_stable floor (B's analogue of
"0/50 viable") and the elite's per-metric h vector (WHERE the half-lives are
lost — replaces the front-size diagnostic).
Decision rule set BEFORE launch, read ONCE. All banked sweeps become re-run
candidates only if B wins (same policy as the aggregation programme).

## Open questions for Luiz

1. steady half-anchor 8.0° (symmetry) — or tighter (e.g. 4°), since steady is
   the headline metric the ladder ranks on?
2. eps floor 2^-20 (h cap 20) — acceptable, or deeper?
3. Does the A/B ride ON the relaunched ladder (2 widths × 2 arms) or run as its
   own small chain first? (Ladder relaunch is already pending the gate-distance
   ranking change + b=10 trim.)
