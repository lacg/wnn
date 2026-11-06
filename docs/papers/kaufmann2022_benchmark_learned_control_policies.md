# Kaufmann, Bauersfeld, Scaramuzza 2022 — A Benchmark Comparison of Learned Control Policies for Agile Quadrotor Flight

Kaufmann, E., Bauersfeld, L., Scaramuzza, D. *A Benchmark Comparison of Learned Control Policies for
Agile Quadrotor Flight.* IEEE ICRA 2022, Philadelphia (pp. 10504–10510 per Swift ref. 44).
arXiv:2202.10796. https://arxiv.org/abs/2202.10796 — Local PDF: `papers/kaufmann2022_benchmark_learned_control_policies.pdf`.
Read 06/11/2026 (full text extracted).

**Relevance.** The canonical action-space comparison for learned quad controllers: SRT (single-rotor
thrusts) vs CTBR (collective thrust + body rates to a low-level rate controller) vs LV (linear velocity
to a full stack). Directly answers our "motor commands vs CTBR cascade" question. **Regime:** PPO, sim
(own TF-Agents sim, 1 ms symplectic Euler) + BEM high-fidelity sim (NeuroBEM) + real 0.768 kg quad;
policies at 50 Hz, classical MPCs at 100 Hz. Teacher is NOT imitation — RL with dense tracking reward.

## Verbatim extraction

Abstract: "training a control policy that commands body-rates and thrust results in more robust
sim-to-real transfer compared to a policy that directly specifies individual rotor thrusts"

Action spaces (Sec. IV): LV `u={vx,vy,vz,ωz}`; CTBR `u={c,ωx,ωy,ωz}` "tracked by a low-level controller
using measurements from an inertial sensor"; SRT `u={f1..f4}` "applied for the duration of a control step".

Why CTBR (Sec. II): "specifying collective thrust and bodyrates does not necessitate estimation of the
full state of the platform, but only requires inertial measurements to perform feedback control on the
bodyrates. This information is readily available at high frequency in today's flight controllers".

Motor model: rotor speeds "modeled as first-order system with time constant kmot where the commanded
motor speeds Ωcmd are the input" (eq. 1). **No numeric kmot is printed and kmot is NOT in the
randomization table.**

Observation: "a history of length H = 10" of z-position, velocity, attitude (rotation matrix), bodyrates
(+ "applied actions" per Sec. IV-A text); reference window R = 10.

Table II (DR, uniform around nominal): mass 0.768 kg ±30%; inertia [2.5e-3, 2.1e-3, 4.3e-3] kg m² ±30%;
gravity ±0.4; kvx/kvy 0.3 N s/m ±0.3; kvz 0.15 ±0.15; cl 1.563e-6 ±0.0; cd 1.909e-8 ±0.0.

Model-mismatch test: BEM simulator + "a control delay of 20 ms to simulate wireless communication latency".

Table V — nominal sim, avg position tracking error [cm], mean±SD over 10 policies:

| Traj | SRT | CTBR | LV | MPC-SRT | MPC-CTBR |
|---|---|---|---|---|---|
| Hover | 1.0±0.2 | 0.6±0.2 | 7.0±1.6 | 0.1 | 0.2 |
| RandA | 1.5±0.2 | 0.9±0.1 | 15.4±3.0 | 0.2 | 0.3 |
| RandB | 2.4±0.2 | 1.6±0.1 | 61.5±21.0 | 0.2 | 0.4 |
| RandC | 3.0±0.3 | 2.0±0.2 | 85.7±11.5 | 0.2 | 0.4 |
| RaceA | 5.0±1.2 | 5.0±1.0 | 121.1±25.8 | 0.3 | 1.3 |
| RaceB | 7.1±1.8 | 6.9±1.5 | 170.2±16.3 | 0.7 | 3.0 |
| Split-S | 3.5±0.4 | 6.6±1.1 | 92.1±20.8 | 1.0 | 2.1 |
| RaceC | 9.2±3.2 | 12.3±2.2 | 197.9±38.1 | 1.2 | 3.9 |

Table VI — BEM sim + 20 ms delay (model mismatch), [cm]:

| Traj | SRT | CTBR | LV | MPC-SRT | MPC-CTBR |
|---|---|---|---|---|---|
| Hover | 11.3±4.5 | 0.6±0.5 | 6.7±2.0 | 1.0 | 0.5 |
| RandA | 12.0±4.0 | 1.2±0.5 | 17.8±1.4 | 3.3 | 1.1 |
| RandB | 14.4±2.4 | 2.2±0.8 | 57.0±12.0 | 7.0 | 2.0 |
| RandC | 17.6±5.9 | 2.6±0.8 | 78.9±13.4 | 8.5 | 2.6 |
| RaceA | crash | 5.6±1.7 | 144.0±20.1 | 12.6 | 4.8 |
| RaceB | crash | 10.0±4.0 | 171.4±17.0 | crash | 6.3 |
| Split-S | crash | 6.9±2.6 | 83.8±9.7 | 11.3 | 11.4 |
| RaceC | crash | 14.9±5.5 | 176.7±22.0 | crash | 7.5 |

Delay (Fig. 4): "policies that operate at higher abstraction levels such as LV or CTBR are less sensitive
to such delay"; "hover is possible for CTBR without a noticeable decrease in performance for latencies up
to 60 ms, the same latency leads to a crash on the racing trajectory."

Real world (Table VII, 5 policies, [cm]): CTBR Hover 4.4±1.4 … Racing 53.0±9.2; LV 6.2±2.0 … 200.8±14.5;
MPC-CTBR 3.0 … 20.0. "Due to its significant sensitivity to control delays and a communication latency of
60 ms imposed by the real system, the SRT policies could not be deployed." CTBR reached "accelerations of
up to 3g and speeds beyond 45 km h−1".

Training (Fig. 3): with DR, SRT learning curves "initially exhibit a high variance, train slower, and
converge to a final performance substantially lower than in case of no domain randomization."

## Absent
- No numeric motor time constant; motor τ not randomized; no motor-lag ablation. The SRT failure is
  attributed to **control delay / model mismatch**, NOT isolated to motor dynamics.
- No imitation-learning arm (all PPO). No SRT policy with action history ablation.
- No low-level rate-controller gains/rate reported for the CTBR inner loop.
- SRT never flown on hardware (so no real SRT number exists).
