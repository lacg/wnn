# Deployment silicon: which chip runs the controller, which runs the IDS

DECISION RECORD. Prices and datasheet figures live in
[`chip_selection_sources.md`](chip_selection_sources.md), every one stamped with its
observation date and source; this file records what we chose and **why**, and is the
one to read first. Both were written 30/08/2026.

## The decision, up front

| workload | chip | why |
|---|---|---|
| **drone controller** | **STM32H743IIT6** (Cortex-M7 MCU) | 1 kHz loop, measured 820 instructions/step. Uses ~0.4% of its time budget and ~17% of its flash. An FPGA is not needed. |
| **IDS** | **XC7Z020 (Zynq Z-7020) FPGA** | Throughput-bound. Every sparse genome we synthesised fits, using **0 BRAM** and ≤62% of LUTs. |

The two workloads want different silicon, and the reason is not model size — it is
**temporal coherence**, which one has and the other does not. Everything below follows
from that.

## Why the same substrate splits

The WNN is identical in both. What differs is the input stream.

A 1 kHz attitude loop has strong temporal coherence: a thermometer level moves a step
or two per millisecond, so exactly one input bit flips. An inverted index turns that
flip into ~16 XORs on a running address instead of a full re-gather, and a neuron whose
address did not change **cannot** change output, so it is never looked up at all. That
is the whole 25x (`mcu/README.md`, [[project_controller_820_instructions_step]]):

```
BASE  full-gather + binsearch   20,231 instr/step
INC   incremental + hash           820 instr/step   <- 25x
```

**Consecutive network flows are unrelated.** There is no coherence to exploit, so an IDS
decision pays the full gather and a lookup for *every* neuron — the BASE path, not the
INC one. Measured 30/08/2026 with `mcu/run_bench_ids.sh` (same QEMU mps2-an386 harness,
same double-subtraction method, fresh independent flow per iteration **on purpose**):

```
shape                    total    gather    search   keys_KB   provenance
64n x 30b  k=1797       17,222     9,985     7,237       449   controller anchor (classify only)
400n x 34b k=256       163,647    70,401    93,246       800   live genome, flows DB
400n x 34b k=1024      173,247    70,401   102,846     3,200   same, deeper memory
500n x 34b k=256       204,547    88,001   116,546     1,000   the shape Vivado synthesised
250n x 100b k=256      234,052   205,800    28,252       500   production winner (>64b, splitmix64)
53n x 48b  k=1024       28,934    13,039    15,895       424   what the GA actually picks
```

Three things that only showed up by measuring:

- **Search dominates the 34-bit shapes**, 93k of 164k — the opposite of the controller,
  where gather leads. 400 neurons each pay a full ~11-iteration binary search over
  64-bit keys, and on a 32-bit core every compare is two words.
- **Deeper memory is nearly free in time, brutal in space.** 4x the keys per neuron costs
  6% more instructions (search is logarithmic) and 4x the flash: 800 KB -> 3,200 KB,
  which no longer fits the H743's 2 MB.
- **The GA's own preferred shape is 8x cheaper.** 53n x 48b at 424 KB is the shape to aim
  at if an MCU IDS is ever wanted — not the production winner.

## The chips we evaluated

All prices observed **30/08/2026**, qty 1 unless stated. Distributor figures are
aggregator-reported (every distributor blocked automated fetch); LCSC fetched directly.
Re-check before quoting — these move.

| | XC7Z020-1CLG400C | XCZU3EG-1SFVC784E | XCZU7EV-2FFVC1156I | STM32H743IIT6 |
|---|---|---|---|---|
| qty 1 | $131.25 | $622.05 | $5,628.48 | $19.95 |
| qty 100 | not found | not found | not found | $12.29 |
| LCSC direct | $81.75 (1+) | — | — | $11.12 (1+), **$8.64 (80+)** |
| LUT6 | 53,200 | 70,560 | 230,400 | n/a |
| BRAM | 4.9 Mb | 7.6 Mb | 11.0 Mb | n/a |
| UltraRAM | none | **0** | 27.0 Mb | n/a |
| PL RAM total | ~6.0 Mb | 9.4 Mb | 44.2 Mb | — |
| flash / SRAM | — | — | — | 2 MB / 1 MB (8.28 Mb) |
| clock | A9 667 MHz | A53 1.5 GHz | A53 1.5 GHz | 480 MHz |
| package | CLG400 17x17 | SFVC784 23x23 | FFVC1156 35x35 | **LQFP176 24x24** |
| **$/Mb of RAM** | **$21.9** | $66.2 | $127.4 | **$2.41** |

### What that table decides

- **The UltraScale+ step is not worth paying.** ZU3EG is +1.9x BRAM for 4.7x the price and
  has **zero UltraRAM** (DS891 Table 3 — it is a BRAM-only step). ZU7EV is 9x the RAM for
  ~43x the price. Both destroy the cheap-hardware framing the WNN work rests on.
- **We do not need to climb that step anyway.** Vivado maps our sparse arrays to **LUT
  distributed RAM, not BRAM** ([[feedback_sparse_fpga_size]]) — every synthesised genome
  reported **0 BRAM**, and UNSW 500n used 33,086 of 53,200 LUTs (62%). For IDS the binding
  resource is LUTs and the Z-7020 has headroom. **Shopping for BRAM is the wrong cliff.**
- **The MCU's 1 MB of SRAM exceeds the Z-7020's 0.62 MB of BRAM** at a seventh of the
  price. For the controller, the FPGA's advantage was never capacity.
- **LQFP176, not BGA.** DS12110 Rev 5 Table 128 decodes I/I/T/6 as 176-pin / 2 MB / LQFP /
  −40..85 °C. Several distributor listings mislabel it UFBGA176. LQFP is hand-solderable,
  which matters for a bench prototype.
- **480 vs 400 MHz is a datasheet revision, not a variant.** Rev 5 (2018) says 400 MHz;
  Rev 7 (2019) says 480 MHz. Quote with the revision.

## Controller on the STM32H743IIT6 — the fit

Memory, from the b=30 footprint work ([[project_controller_lut_footprint_b30]]):

| winner | keys | packed | vs 2 MB flash |
|---|---|---|---|
| lqi s31337002 | 115,023 | 336 KB | 17% |
| lqi s31337003 | 167,337 | 512 KB | 26% |
| mpc s31337002 | 105,906 | 319 KB | 16% |

One wrinkle: the 820-instruction figure used an open-addressed hash at HBITS=18 costing
**2 MB of RAM**, and this part has 1 MB of SRAM. That implementation does not fit — but it
does not need to. Keep incremental addressing (which bought the 25x) and use binary search
over sorted keys in flash: only the few neurons whose address changed are looked up, ~13
iterations each. Call it ~1,800 instructions/step, roughly 2-4 µs at 480 MHz against a
1 ms period — **under half a percent of budget.** A minimal perfect hash would cut it
further and is the obvious next step.

## Why 10 MCUs do not rescue the IDS

Ten chips do **not** give 10x throughput *and* 4x memory. They are different architectures
and you pick one:

- **Shard the flows, replicate the model** — each chip holds the *whole* model, handles
  1/10 of traffic. 10x throughput; the memory ceiling stays **2 MB per chip**. Buying more
  chips never enlarges the model you can hold.
- **Shard the neurons, broadcast the flow** — chip *k* holds 50 of 500 neurons, all score
  the same input, partial vote counts are summed. This *does* aggregate memory to ~20 MB,
  and it works because WNN neurons are genuinely independent and the response is a sum
  over them. But every chip works on every flow, so throughput does not multiply, and an
  interconnect lands on the critical path.

At ~$12-20 each, 10 chips is $120-200 — the same order as one Z-7020 at $131, for
throughput still ~200x short. **The economics do not favour the array either.**

## Open — do not quote these as settled

- **Real keys-per-neuron for a trained IDS genome is UNKNOWN.** `genomes.materialized_cells`
  is unpopulated for `architecture_type='ids'` (0 rows), so the sweep above varies it
  instead of measuring it. The controller sees 1,797/neuron at b=30 on far less data; an
  IDS genome trained on 46M flows is plausibly 10k+, which would put 400n x 34b at ~32 MB
  and off the chip entirely regardless of speed. **Train-and-count before quoting an MCU
  IDS footprint** — the dense formula is fiction ([[feedback_sparse_fpga_size]]).
- **Earlier price figures do not reproduce.** A previous discussion recorded XC7Z020 at
  $24.32, XCZU3EG at $707.20 and the MCU near $7.35. Of these only ZU3EG is close
  ($622.05). The cheapest XC7Z020 found is LCSC $81.75, and the MCU's $7-ish only appears
  at LCSC's 80+ break ($8.64). Prices move and the earlier source is unrecoverable — trust
  the stamped table, and re-check before publishing.
- XC7Z020 distributed-RAM total is **not published** by AMD (DS190/XMP097/UG474 all omit a
  Zynq row); the ~6.0 Mb PL total uses a labelled derivation, not a quote.
- ZU7EV authorized stock is **0** at all three distributors; brokers hold volume at
  $338-345 vs $5,628 list. Do not treat broker pricing as market price.
- PYNQ-Z2 shows a 2.9x spread across authorized distributors ($99.47 Arrow vs $284.90
  DigiKey). Suspect; re-check Arrow directly.

## Reproducing the measurements

```bash
N=100 bash mcu/run_bench_ids.sh      # IDS decision cost across genome shapes
N=200 bash mcu/run_bench.sh          # controller step cost, WNN vs PID vs MLP
```

Needs `brew install arm-none-eabi-gcc qemu`. Both count **retired instructions** under
QEMU, not silicon cycles — mps2-an386 does not implement `DWT_CYCCNT`. Real cycles run
above these counts (loads ~2, taken branches ~2-3, flash waits). Treat them as a
like-for-like ratio and a lower bound, never as a timing claim. Hardware DWT numbers come
with the flight test.

Synthetic keys are legitimate for a *cost* measurement because the search runs to
convergence with no early exit, so a hit and a miss cost identically — the bounded-WCET
property the FPGA claim already rests on. They say **nothing** about accuracy.
