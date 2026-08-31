# Chip selection: verified prices and on-chip memory

Deployment-target survey for the two WNN workloads:
**(A) drone attitude controller** — 1 kHz, tiny, latency-trivial, *memory-bound* (sparse RAM-neuron
tables; see `project_controller_lut_footprint_b30`, b=30 fits Z-7020) and
**(B) network IDS classifier** — *throughput-bound* (wide N-tuple ensembles, streaming).

**All prices observed 30/08/2026.** Prices are volatile; re-stamp before any publication.
Every number below is copied from a page that was actually fetched in this run. Cells that could
not be fetched read **not found** — they are NOT estimated. See §6 for the full unverified list.

## 0. Sourcing caveat (read before quoting any price)

Digi-Key, Mouser, Newark, Arrow and Octopart **all refused automated fetch**
(HTTP 403 / connection reset) on 30/08/2026. Distributor-attributed prices below therefore come
from the **oemstrade aggregator** (`https://www.oemstrade.com`), which names the distributor and
its price break next to each line. That is a *secondary* source: it attributes to DigiKey/Mouser/
Newark but was not read off the distributor's own page. LCSC pages fetched directly.
Treat authorized-distributor rows as "aggregator-reported, distributor-attributed", and broker
rows as **broker, not list price**.

---

## 1. Numbers first

- **On-chip RAM available to a WNN, per part** (PL memories only, MCU = SRAM):
  Z-7020 **≈4.9 Mb BRAM (0.62 MB)** · ZU3EG **7.6 Mb BRAM + 1.8 Mb distributed = 9.4 Mb (1.17 MB)** ·
  ZU7EV **11.0 Mb BRAM + 27.0 Mb UltraRAM + 6.2 Mb distributed = 44.2 Mb (5.52 MB)** ·
  STM32H743II **1 MB SRAM (8.28 Mb incl. backup) + 2 MB flash (16 Mb)**.
- **ZU7EV has ~9x the Z-7020's on-chip RAM** (44.2 Mb vs 4.9 Mb PL RAM) and **4.3x the LUT6 count**
  (230,400 vs 53,200) — but costs **~43x** at qty 1 ($5,628.48 DigiKey vs $131.25 DigiKey).
- **ZU3EG is the only FPGA step that buys RAM without a price cliff**: +1.9x BRAM over Z-7020 for
  **4.7x** the price ($622.05 vs $131.25, DigiKey qty 1). It has **zero UltraRAM**.
- **The MCU is the cheap outlier**: STM32H743IIT6 **$17.71 qty 1 / $12.29 qty 100** (Mouser via
  aggregator) — **~7x cheaper than the Z-7020 chip alone**, and its 1 MB SRAM *exceeds* the
  Z-7020's 0.62 MB of BRAM. For workload (A) (820 instr/step measured on Cortex-M4) this part is
  not obviously worse on memory; it loses only on parallel throughput, i.e. workload (B).
- **Realistic FPGA buy is a board, not a chip.** PYNQ-Z2 (DFR0600) **$284.90 DigiKey / $99.47
  Arrow** (a 2.9x spread on the same SKU — verify before quoting); ZCU104 **$1,899.12 DigiKey**.
  A board more than doubles the Z-7020 bill of materials and more than *triples* nothing on ZU7EV
  (board $1,899 is *cheaper* than the bare XCZU7EV die at $5,628).

---

## 2. Comparison table

| | **XC7Z020-1CLG400C** (Z-7020) | **XCZU3EG-1SFVC784E** | **XCZU7EV-2FFVC1156I** | **STM32H743IIT6** |
|---|---|---|---|---|
| Class | Zynq-7000 SoC (28 nm) | Zynq UltraScale+ MPSoC EG | Zynq UltraScale+ MPSoC EV | Cortex-M7 MCU |
| **Price qty 1 (USD)** | **$131.25** DigiKey; $131.25 Mouser; $128.63 Newark | **$622.05** DigiKey; $604.43 Mouser; $578.52 Newark | **$5,628.48** DigiKey; $5,628.49 Mouser; $5,237.22 Newark | **$19.95** DigiKey; **$17.71** Mouser; $17.69 Newark; $9.69–$12.57 Arrow |
| **Price qty 10** | $123.48 Newark | not found | $5,027.73 Newark | $15.21 Mouser; $14.05 Newark |
| **Price qty 100** | not found (Newark table stops at 25 → $120.91) | not found | not found (Newark stops at 25 → $4,922.99) | $12.29 Mouser; $12.40 Newark |
| Direct-fetched alt. | LCSC $81.7481 (1+), $78.1204 (30+) — LCSC is not an AMD-authorized franchise; treat as reseller | not found | not found | LCSC $11.1152 (1+), $10.0171 (10+), $9.3148 (40+), $8.6354 (80+) |
| Authorized stock (30/08/2026) | DigiKey 0, Mouser 0, Newark 0 (LCSC 170) | DigiKey 1 tray, Mouser 2, Newark 0 | DigiKey 0, Mouser 0, Newark 0 | DigiKey 1,919 trays; Mouser 1,258; Newark 4,750 |
| Lifecycle | Active (no EOL notice found); **AMD 7-series 2035 longevity claim NOT verified** | Active (no EOL notice found) | Active (no EOL notice found) — **at qty1 authorized stock 0, only brokers hold volume** | Full production (DS12110) |
| **LUT6 count** | **53,200** | **70,560** | **230,400** | n/a |
| Flip-flops | 106,400 | 141,120 | 460,800 | n/a |
| **Block RAM** | **4.9 Mb** (140 × 36 Kb) = **0.62 MB** | **7.6 Mb** (216 blocks) = **0.95 MB** | **11.0 Mb** (312 blocks) = **1.37 MB** | n/a |
| **UltraRAM** | none (architecture) | **0 Mb** (0 blocks) | **27.0 Mb** (96 × 288 Kb) = **3.38 MB** | n/a |
| **Distributed / LUT RAM** | **not found** in AMD docs — derived ≈1.08 Mb ≈ 0.14 MB (see §4) | **1.8 Mb** = **0.23 MB** | **6.2 Mb** = **0.78 MB** | n/a |
| PL RAM total | ≈6.0 Mb ≈ 0.75 MB (incl. derived distRAM) | **9.4 Mb = 1.17 MB** | **44.2 Mb = 5.52 MB** | n/a |
| PS on-chip mem | 256 KB OCM | 256 KB OCM w/ECC | 256 KB OCM w/ECC | — |
| **Flash** | none on-die | none on-die | none on-die | **2 MB = 16 Mb** |
| **SRAM** | — | — | — | **1 MB** = 192 KB TCM (64 ITCM + 128 DTCM) + up to 864 KB user SRAM + 4 KB backup = 1,060 KB = **8.28 Mb** |
| DSP | 220 | 360 | 1,728 | n/a |
| **Max clock** | PS Cortex-A9 **667 MHz (-1)** / 766 MHz (-2) / 866 MHz (-3). PL Fmax design-dependent. | APU Cortex-A53 **up to 1.5 GHz**, RPU Cortex-R5F up to 600 MHz. PL Fmax design-dependent. | same PS as ZU3EG (up to 1.5 GHz / 600 MHz) | **up to 480 MHz** (DS12110 Rev 7); Rev 5 stated 400 MHz |
| Speed grade in the P/N | **-1** (commercial, `C`) | **-1**, extended (`E`) | **-2**, industrial (`I`) | n/a |
| **Package** | CLG400, **17 × 17 mm**, 0.8 mm ball pitch, 400-ball BGA | SFVC784, **23 × 23 mm**, 0.8 mm pitch | FFVC1156, **35 × 35 mm**, 1.0 mm pitch | **LQFP176, 24 × 24 mm**, 0.5 mm pitch |
| Bare chip or board? | **Board in practice** — PYNQ-Z2 / Zybo Z7-20 | **Board in practice** — Ultra96-V2 (ZU3EG) | **Board in practice** — ZCU104 | **Bare chip is realistic** (LQFP, hand-solderable-ish; Nucleo/eval also exists) |
| Representative board | **PYNQ-Z2 (DFR0600)**: $284.90 DigiKey, $323.75 Mouser, **$99.47 Arrow/Verical** | **Ultra96-V2 (AES-ULTRA96-V2-G)**: **not found** (all listings RFQ/broker) | **ZCU104 (EK-U1-ZCU104-G)**: **$1,899.12 DigiKey**, $1,877.62 Mouser, $1,795.35 Newark | n/a |

### Ordering-code decode (primary source, ST DS12110 Rev 5 Table 128)

`STM32H743 I I T 6` → **I = 176 pins/balls · I = 2 Mbytes flash · T = LQFP · 6 = industrial, −40 to 85 °C**.
(So the `T6` part is **LQFP176, not BGA** — several distributor listings mislabel it UFBGA176.)

---

## 3. Verbatim extractions

### 3.1 AMD DS190, *Zynq-7000 SoC Data Sheet: Overview*, v1.11.1, July 2, 2018 — Table 1
Z-7020 / **XC7Z020** column, quoted exactly:

| Field | Value |
|---|---|
| 7 Series PL Equivalent | `Artix-7 FPGA` |
| Programmable Logic Cells | `85K` |
| Look-Up Tables (LUTs) | `53,200` |
| Flip-Flops | `106,400` |
| Block RAM (# 36 Kb Blocks) | `4.9 Mb` `(140)` |
| DSP Slices (18x25 MACCs) | `220` |
| Peak DSP Performance (Symmetric FIR) | `276 GMACs` |
| Maximum Frequency | `667 MHz (-1); 766 MHz (-2); 866 MHz (-3)` |
| Processor Core | `Dual-core ARM Cortex-A9 MPCore with CoreSight` |
| On-Chip Memory (PS) | `256 KB` |
| L2 Cache | `512 KB` |

DS190 Table 2, CLG400 package: `Size 17 x 17 mm`, `Ball Pitch 0.8 mm`, XC7Z020 row `128` PS I/O,
SelectIO `125` HR, `–` HP.
XMP097 (Zynq-7000 SoC Product Selection Guide, © 2014–2019, page 2) Speed Grades row for Z-7020:
Commercial `-1`; Extended `-2,-3`; Industrial `-1, -2, -1L`.

**Absent from DS190 and XMP097:** any *Distributed RAM* / LUT-RAM column for Zynq-7000. Neither
document lists it. See §4.

### 3.2 AMD DS891, *Zynq UltraScale+ MPSoC Data Sheet: Overview*, v1.11.1, March 18, 2025

Table 3 (EG Device Feature Summary), **ZU3EG** column, quoted exactly:

| Field | ZU3EG |
|---|---|
| System Logic Cells | `154,350` |
| CLB Flip-Flops | `141,120` |
| CLB LUTs | `70,560` |
| Distributed RAM (Mb) | `1.8` |
| Block RAM Blocks | `216` |
| Block RAM (Mb) | `7.6` |
| UltraRAM Blocks | `0` |
| UltraRAM (Mb) | `0` |
| DSP Slices | `360` |
| Max. HP I/O | `156` |
| Max. HD I/O | `96` |

Table 5 (EV Device Feature Summary), **ZU7EV** column, quoted exactly:

| Field | ZU7EV |
|---|---|
| Video Codec | `1` |
| System Logic Cells | `504,000` |
| CLB Flip-Flops | `460,800` |
| CLB LUTs | `230,400` |
| Distributed RAM (Mb) | `6.2` |
| Block RAM Blocks | `312` |
| Block RAM (Mb) | `11.0` |
| UltraRAM Blocks | `96` |
| UltraRAM (Mb) | `27.0` |
| DSP Slices | `1,728` |
| Max. HP I/O | `416` |
| Max. HD I/O | `48` |

DS891 p.1: APU `CPU frequency: Up to 1.5GHz`; RPU `CPU frequency: Up to 600MHz`;
On-Chip Memory `256KB on-chip RAM (OCM) in PS with ECC`, `Up to 36Mb on-chip RAM (UltraRAM) with
ECC in PL`, `Up to 35Mb on-chip RAM (block RAM) with ECC in PL`, `Up to 11Mb on-chip RAM
(distributed RAM) in PL` (family maxima, **not** ZU3EG/ZU7EV).
DS891 p.3: block RAM is `36Kb Block RAM`, `True dual-port`, `Up to 72 bits wide`,
`Configurable as dual 18Kb`. UltraRAM is `288Kb dual-port`, `72 bits wide`, `Error checking and
correction`.
Table 4: ZU3EG packages `UBVA530 (9.5x16)`, `SBVA484 (19x19)`, `SFVA625 (21x21)`, `SFVC784 (23x23)`.
Table 6: ZU7EV packages `FBVB900 (31x31)`, `FFVC1156 (35x35)`, `FFVF1517 (40x40)` —
**ZU7EV is NOT offered in SFVC784 or SFVE784** (those cells are greyed out).
Note 2: `FB/FF packages have 1.0mm ball pitch. SF packages have 0.8mm ball pitch.`

### 3.3 Xilinx UG474, *7 Series FPGAs Configurable Logic Block User Guide*, v1.7, Nov 17, 2014
- `Real 6-input look-up table (LUT) technology` — **confirms LUT6** for the Z-7020's Artix-7-class PL.
- `Approximately two-thirds of the slices are SLICEL logic slices and the rest are SLICEM, which
  can also use their LUTs as distributed 64-bit RAM or as 32-bit shift registers (SRL32) or as two
  SRL16s.`
- `A 6-input LUT can be used as a 64 x 1 memory for small storage requirements.`
- **Absent:** Tables 1-1/1-2/1-3 cover Artix-7, Kintex-7 and Virtex-7 only. **No Zynq-7000 row.**
  This is why the XC7Z020 distributed-RAM total is "not found" in primary AMD documentation.

### 3.4 ST DS12110, *STM32H742xI/G STM32H743xI/G*, Rev 7, April 2019 (cover page)
- Title: `32-bit Arm Cortex-M7 480MHz MCUs, up to 2MB Flash, up to 1MB RAM, 46 com. and analog interfaces`
- Core: `32-bit Arm Cortex-M7 core with double-precision FPU and L1 cache: 16 Kbytes of data and
  16 Kbytes of instruction cache; frequency up to 480 MHz, MPU, 1027 DMIPS/2.14 DMIPS/MHz
  (Dhrystone 2.1), and DSP instructions`
- Memories: `Up to 2 Mbytes of Flash memory with read-while-write support`;
  `Up to 1 Mbyte of RAM: 192 Kbytes of TCM RAM (inc. 64 Kbytes of ITCM RAM + 128 Kbytes of DTCM RAM
  for time critical routines), Up to 864 Kbytes of user SRAM, and 4 Kbytes of SRAM in Backup domain`
- Packages shown: `LQFP100 (14 x 14 mm)`, `LQFP144 (20 x 20 mm)`, `LQFP176 (24 x 24 mm)`,
  `LQFP208 (28 x 28 mm)`, `TFBGA100 (8 x 8 mm)`, `TFBGA240+25 (14 x 14 mm)`, `UFBGA169 (7 x 7 mm)`,
  `UFBGA176+25 (10 x 10 mm)`

**Regime note (rev drift):** DS12110 **Rev 5** (July 2018), same document, says
`frequency up to 400 MHz, MPU, 856 DMIPS`. The 480 MHz figure is Rev 7+ (revision-V silicon).
Newark's listing still describes the part as `MCU, 32BIT, 400MHZ`. If a paper quotes 480 MHz it
must also state the silicon revision.

ST DS12110 Rev 5, Table 127 (thermal): `LQFP176 - 24 x 24 mm /0.5 mm pitch`, θJA `43.0 °C/W`.
Table 128 ordering scheme: pin count `I = 176 pins/balls`; flash `I = 2 Mbytes`;
package `T = LQFP`, `K = UFBGA pitch 0.65 mm`, `I = UFBGA pitch 0.5 mm`, `H = TFBGA`;
temperature `6 = Industrial temperature range, –40 to 85 °C`.

### 3.5 Distributor listings (all observed 30/08/2026)

**XC7Z020-1CLG400C** — via oemstrade:
Newark stock 0: `1x $128.63 | 5x $126.06 | 10x $123.48 | 25x $120.91`.
Mouser stock 0: `1x $131.25`. DigiKey stock 0 (Tray): `1x $131.25`.
Brokers (**not list price**): ICHOME 1,000 @ $136.71; Bristol 1 @ $126.00; Win Source 8,413/3,500,
`1x $95.33`; IC Components 16,527 @ `1x $21.90`; Quest Components tiers `$93.71–$393.75`.
Direct fetch, LCSC (reseller, not AMD-franchised): `1+ $81.7481`, `30+ $78.1204`, stock 170.

**XCZU3EG-1SFVC784E** — via oemstrade:
Newark stock 0 `$578.52 (1x)`; Mouser stock 2 `$604.43 (1x)`; DigiKey stock 1 Tray `$622.05 (1x)`.
Brokers: Component Stockers 3 @ $554.12; IC Components 10,100 @ $114.88;
Win Source 900 @ `$124.00 (1x), $119.34 (2x), $114.68 (3x)`.
*The ~5x gap between authorized (~$600) and broker (~$115–124) is the usual counterfeit/grey-market
tell — do not quote broker rows as market price.*

**XCZU7EV-2FFVC1156I** — via oemstrade:
Newark stock 0: `1 unit @ $5,237.22; 5 @ $5,132.48; 10 @ $5,027.73; 25 @ $4,922.99`.
Mouser stock 0: `1 unit @ $5,628.49`. DigiKey stock 0: `1 unit @ $5,628.48`.
Brokers: ICHOME 1,000 @ $4,737.96; IC Components 2,590 @ $338.83; Win Source 360 @ $344.70.

**STM32H743IIT6** — via oemstrade:
Newark 4,750: `1x $17.69 | 10x $14.05 | 25x $13.41 | 50x $12.76 | 100x $12.40 | 250x $12.04 | 500x $11.62`
Mouser 1,258: `1x $17.71 | 10x $15.21 | 25x $12.53 | 100x $12.29 | 250x $12.22 | 400x $11.43`
DigiKey 1,919 trays: `1x $19.95 | 40x $14.40 | 80x $13.86 | 120x $13.58 | 520x $12.83`
Arrow/Verical 5,941: `1x $9.69–$12.57 | 40x $9.98–$9.99`
Future Electronics stock 0, lead time 30 weeks: `3x $10.27 | 5x $10.22 | 20x $10.07 | 30x $10.03 | 75x $9.89`
Direct fetch, LCSC: `1+ $11.1152 | 10+ $10.0171 | 40+ $9.3148 | 80+ $8.6354`, stock 7,
package `LQFP-176(24x24)`, Flash `2MB`, RAM `Up to 1MB (including 192KB TCM RAM, up to 864KB user
SRAM, 4KB backup SRAM)`, `480MHz`.

**Boards** — via oemstrade:
`EK-U1-ZCU104-G` (ZCU104, XCZU7EV): Newark 81 @ $1,795.35; Mouser 109 @ $1,877.62;
DigiKey 138 @ $1,899.12; Component Stockers 119 @ $1,803.31; (brokers $2,979.70–$3,065.26).
`DFR0600` (PYNQ-Z2, XC7Z020): Mouser 121 @ $323.75; DigiKey 153 @ $284.90;
Arrow 38 @ $99.47; Verical 38 @ $99.47; Neutron USA 50 @ $573.03.
**Flag:** a 2.9x spread on one SKU across authorized distributors is anomalous — Arrow's $99.47
may be a stale/mis-mapped line. Do not publish $99.47 without re-checking Arrow's own page.
`AES-ULTRA96-V2-G` (Ultra96-V2, ZU3EG): **all listings RFQ / "OEM/CM QUOTES ONLY | NO BROKERS"** —
**no price found**.

PYNQ project board list (pynq.io/boards.html) confirms `PYNQ-Z2: Zynq Z7020` and that `ZCU104`,
`AUP-ZU3`, `Ultra96V2`, `ZUBoard 1CG`, `TySOM-3-ZU7EV` are the supported UltraScale+ boards.
No prices on that page.

---

## 4. Derivations (ours, not quoted)

All Mb/MB below use **binary** units (1 Mb = 1,048,576 bits; 1 MB = 1,048,576 bytes), which is what
makes AMD's own "140 × 36 Kb = 4.9 Mb" arithmetic close. Formula: `MB = bits / 8 / 1048576`.

| Quantity | Formula | Result |
|---|---|---|
| Z-7020 BRAM | 140 × 36 × 1024 = 5,160,960 bit | 4.92 Mb → **0.615 MB (630 KiB)** |
| Z-7020 distributed RAM (**estimate, not a quote**) | UG474: "approximately two-thirds SLICEL, the rest SLICEM"; SLICEM LUTs = 64 bit each → (53,200 / 3) × 64 = 1,134,933 bit | ≈1.08 Mb → **≈0.135 MB** — **AMD publishes no figure for Zynq-7000; treat as ±30%** |
| ZU3EG BRAM | 216 × 36 × 1024 = 7,962,624 bit | 7.59 Mb → **0.949 MB (972 KiB)** |
| ZU3EG distributed RAM | 1.8 Mb × 1,048,576 / 8 | **0.225 MB (230 KiB)** |
| ZU7EV BRAM | 312 × 36 × 1024 = 11,501,568 bit | 10.97 Mb → **1.371 MB** |
| ZU7EV UltraRAM | 96 × 288 × 1024 = 28,311,552 bit | 27.00 Mb → **3.375 MB** |
| ZU7EV distributed RAM | 6.2 Mb × 1,048,576 / 8 | **0.775 MB** |
| ZU3EG PL RAM total | 7.59 + 1.80 | **9.39 Mb = 1.174 MB** |
| ZU7EV PL RAM total | 10.97 + 27.00 + 6.20 | **44.17 Mb = 5.521 MB** |
| STM32H743 flash in Mb | 2 MB × 8 | **16 Mb** |
| STM32H743 SRAM in Mb | (192 + 864 + 4) KB = 1,060 KB × 8 × 1024 / 1,048,576 | **8.28 Mb** (ST rounds 1,060 KB to "1 Mbyte") |

Cost-per-bit at qty 1 (our derivation, authorized DigiKey price ÷ total on-chip WNN-usable RAM):
Z-7020 $131.25 / 6.0 Mb = **$21.9/Mb** · ZU3EG $622.05 / 9.4 Mb = **$66.2/Mb** ·
ZU7EV $5,628.48 / 44.2 Mb = **$127.4/Mb** · STM32H743IIT6 $19.95 / 8.28 Mb SRAM = **$2.41/Mb**
(or $19.95 / 24.28 Mb incl. flash = **$0.82/Mb**). *Flash is not writable at 1 kHz — for a
learning WNN only the SRAM/BRAM columns count.*

---

## 5. Sources

- AMD, *Zynq-7000 SoC Data Sheet: Overview*, DS190 (v1.11.1), July 2, 2018 —
  <https://docs.amd.com/api/khub/documents/juMnxca71Tf2gfjmNyjM8A/content>
  (local: `papers/xilinx2018_ds190_zynq7000_overview.pdf`)
- AMD, *Zynq UltraScale+ MPSoC Data Sheet: Overview*, DS891 (v1.11.1), March 18, 2025 —
  <https://docs.amd.com/api/khub/documents/sbPbXcMUiRSJ2O5STvuGNQ/content>
  (local: `papers/amd2025_ds891_zynq_ultrascale_plus_overview.pdf`)
- Xilinx, *Zynq-7000 SoC Product Selection Guide*, XMP097 (v1.3.2) —
  <https://docs.amd.com/api/khub/documents/1L_hkh2pbc5l0Oz7tcLspA/content>
  (local: `papers/xilinx2019_xmp097_zynq7000_selection_guide.pdf`)
- Xilinx, *7 Series FPGAs Configurable Logic Block User Guide*, UG474 (v1.7), Nov 17, 2014 —
  <https://www.eng.auburn.edu/~nelson/courses/elec4200/FPGA/ug474_7Series_CLB.pdf>
  (local: `papers/xilinx2014_ug474_7series_clb.pdf`)
- ST, *STM32H742xI/G STM32H743xI/G datasheet*, DS12110 Rev 7, April 2019 —
  <https://www.tme.eu/Document/537e4e89f9733480322b3adb059e9604/STM32H743BIT6.pdf>
  (local: `papers/st2019_ds12110_rev7_stm32h742_h743.pdf`)
- ST, *STM32H743xI datasheet*, DS12110 Rev 5, July 2018 (ordering scheme, thermal) —
  <https://cdn.sparkfun.com/assets/e/c/9/c/6/STM32H743VI.pdf>
  (local: `papers/st2018_ds12110_rev5_stm32h743xi.pdf`)
- oemstrade aggregator, observed 30/08/2026 —
  <https://www.oemstrade.com/search/xc7z020-1clg400c> ·
  <https://www.oemstrade.com/search/xczu3eg-1sfvc784e> ·
  <https://www.oemstrade.com/search/xczu7ev-2ffvc1156i> ·
  <https://www.oemstrade.com/search/stm32h743iit6> ·
  <https://www.oemstrade.com/search/ek-u1-zcu104-g> ·
  <https://www.oemstrade.com/search/dfr0600> ·
  <https://www.oemstrade.com/search/ultra96-v2>
- LCSC (direct fetch, 30/08/2026) — <https://www.lcsc.com/product-detail/C2988113.html> ·
  <https://www.lcsc.com/product-detail/ST-Microelectronics_STMicroelectronics-STM32H743IIT6_C89597.html>
- PYNQ supported boards — <http://www.pynq.io/boards.html>

---

## 6. NOT verified — do not cite these

- **Qty-100 price for all three FPGAs.** Newark's break tables stop at 25; DigiKey/Mouser show only
  qty 1. Not found.
- **Qty-10 price for XCZU3EG-1SFVC784E.** Only qty 1 was listed anywhere. Not found.
- **XC7Z020 distributed/LUT-RAM total.** No AMD document (DS190, XMP097, UG474) publishes it for
  Zynq-7000. The §4 figure is OUR estimate from the UG474 SLICEL/SLICEM ratio, not a quote.
- **Ultra96-V2 (AES-ULTRA96-V2-G) board price.** Every listing was RFQ/quote-only. Not found.
- **Any price read off a Digi-Key, Mouser, Newark, Arrow or Octopart page.** All five refused
  automated fetch (403 / ECONNRESET) on 30/08/2026. Every "DigiKey/Mouser/Newark/Arrow" figure here
  is aggregator-reported.
- **AMD's "7 series supported until at least 2035" longevity claim.** A search snippet asserted it;
  the AMD support page returned a loading error and could not be read. Not verified.
- **Lifecycle status strings** (Active / NRND / LTB) for all four parts — no distributor page could
  be opened to read the status field. "Active" above is inferred from continued authorized listing
  + stock, not from a status field. No EOL/NRND notice was found for any of the four.
- **PL Fmax** for all three FPGAs — design-dependent, not a datasheet number.
- **PYNQ-Z2 Arrow price ($99.47)** — 2.9x below DigiKey/Mouser for the same SKU; unverified against
  Arrow's own page and possibly a stale line.
- **Speed-grade availability tables for ZU3EG/ZU7EV** (which of -1/-2/-3 ship in which package) —
  DS891 Tables 4/6 give packages and I/O only, not speed-grade availability.
- **XCZU3EG / XCZU7EV maximum PS frequency at these specific speed grades** — DS891 p.1 gives family
  maxima ("up to 1.5GHz") only; per-speed-grade PS Fmax is in the AC/DC switching characteristics
  datasheets (DS925/DS926), which were not fetched.

---

# 7. More LUTs than the Z-7020 at the same money? — Spartan-7 / Artix-7 survey

**All prices in this section observed 30/08/2026.** Same sourcing caveat as §0: Digi-Key,
Mouser, Newark, Arrow and Octopart still refuse automated fetch, so distributor-attributed
prices come from the **oemstrade** aggregator. **LCSC rows in this section were fetched
directly from LCSC product pages** and are the stronger evidence.

## 7.0 What changed vs §1–§6, and what the binding resource actually is

§1–§6 were written assuming BRAM was the target. The IDS synthesis reports say otherwise.
All 21 Vivado utilization reports for the real IDS designs read:

```
LUT as Logic     50527  (94.98%)
LUT as Memory        0  (0.00%   of 17400 available)
Block RAM Tile       0  (0.00%   of 140 available)
```

So the shipped designs are **100% combinational logic** — the RAM-neuron contents are
constant-folded into gates. **Distributed RAM (SLICEM) and block RAM are both entirely
unused.** Therefore:

- **Primary metric = LUT6 per dollar.** (Not distributed RAM. A distributed-RAM ranking was
  dropped from this section; the DS180 distributed-RAM column is retained in the table below
  as reference data only, not as a ranking key.)
- **Secondary metric = block RAM Mb per dollar**, because we have a proven alternative
  representation — sparse keys + binary search — that on the *controller* converted a design
  needing 106% of LUTs into 55% of BRAM on the same part
  (`project_controller_lut_footprint_b30`). If that representation is adopted for the IDS,
  BRAM becomes binding instead of LUTs, so both columns must be evaluated against one part list.

**Real design footprints on the Z-7020 (53,200 LUT6):** 1,939 LUT (UNSW 92n) up to
**50,527 LUT (CICIOT46M 500n × 34b, 94.98%)**. Two separate designs sit at ~95%. We are **at
the ceiling on this part**, not comfortably inside it.

## 7.1 Verbatim: AMD DS180, *7 Series FPGAs Data Sheet: Overview*, v2.5, August 1, 2017

### Table 2: Spartan-7 FPGA Feature Summary by Device (quoted exactly)

| Device | Logic Cells | Slices | Max Distributed RAM (Kb) | DSP Slices | BRAM 18 Kb | BRAM 36 Kb | BRAM Max (Kb) | CMTs | PCIe | GT | XADC | I/O Banks | Max User I/O |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| XC7S6 | `6,000` | `938` | `70` | `10` | `10` | `5` | `180` | `2` | `0` | `0` | `0` | `2` | `100` |
| XC7S15 | `12,800` | `2,000` | `150` | `20` | `20` | `10` | `360` | `2` | `0` | `0` | `0` | `2` | `100` |
| XC7S25 | `23,360` | `3,650` | `313` | `80` | `90` | `45` | `1,620` | `3` | `0` | `0` | `1` | `3` | `150` |
| **XC7S50** | `52,160` | `8,150` | `600` | `120` | `150` | `75` | `2,700` | `5` | `0` | `0` | `1` | `5` | `250` |
| **XC7S75** | `76,800` | `12,000` | `832` | `140` | `180` | `90` | `3,240` | `8` | `0` | `0` | `1` | `8` | `400` |
| **XC7S100** | `102,400` | `16,000` | `1,100` | `160` | `240` | `120` | `4,320` | `8` | `0` | `0` | `1` | `8` | `400` |

### Table 4: Artix-7 FPGA Feature Summary by Device (quoted exactly)

| Device | Logic Cells | Slices | Max Distributed RAM (Kb) | DSP48E1 | BRAM 18 Kb | BRAM 36 Kb | BRAM Max (Kb) | CMTs | PCIe | GTPs | XADC | I/O Banks | Max User I/O |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| XC7A12T | `12,800` | `2,000` | `171` | `40` | `40` | `20` | `720` | `3` | `1` | `2` | `1` | `3` | `150` |
| XC7A15T | `16,640` | `2,600` | `200` | `45` | `50` | `25` | `900` | `5` | `1` | `4` | `1` | `5` | `250` |
| XC7A25T | `23,360` | `3,650` | `313` | `80` | `90` | `45` | `1,620` | `3` | `1` | `4` | `1` | `3` | `150` |
| **XC7A35T** | `33,280` | `5,200` | `400` | `90` | `100` | `50` | `1,800` | `5` | `1` | `4` | `1` | `5` | `250` |
| **XC7A50T** | `52,160` | `8,150` | `600` | `120` | `150` | `75` | `2,700` | `5` | `1` | `4` | `1` | `5` | `250` |
| XC7A75T | `75,520` | `11,800` | `892` | `180` | `210` | `105` | `3,780` | `6` | `1` | `8` | `1` | `6` | `300` |
| **XC7A100T** | `101,440` | `15,850` | `1,188` | `240` | `270` | `135` | `4,860` | `6` | `1` | `8` | `1` | `6` | `300` |
| **XC7A200T** | `215,360` | `33,650` | `2,888` | `740` | `730` | `365` | `13,140` | `10` | `1` | `16` | `1` | `10` | `500` |

**DS180 Note 1, verbatim (both tables):**
`Each 7 series FPGA slice contains four LUTs and eight flip-flops; only some slices can use
their LUTs as distributed RAM or SRLs.`
**DS180 Note 3, verbatim:** `Block RAMs are fundamentally 36 Kb in size; each block can also
be used as two independent 18 Kb blocks.`

DS180 does **not** print a "LUTs" column — it prints **Slices**. The LUT6 counts used for
ranking below are OUR derivation, `LUTs = Slices × 4`, straight from Note 1 (see §7.4).

### Package tables (quoted exactly)

**Table 3, Spartan-7 device-package combinations:**
`CPGA196` `8x8` `0.5` · `CSGA225` `13 x 13` `0.8` · `CSGA324` `15 x 15` `0.8` ·
`FTGB196` `15 x 15` `1.0` · `FGGA484` `23 x 23` `1.0` · `FGGA676` `27 x 27` `1.0`.
XC7S50 rows: `CSGA324 210`, `FTGB196 100`, `FGGA484 250`. XC7S75/XC7S100 rows:
`FGGA484 338`, `FGGA676 400` only.

**Table 5, Artix-7 device-package combinations:**
`CPG236` `10 x 10` `0.5` · `CPG238` `10 x 10` `0.5` · `CSG324` `15 x 15` `0.8` ·
`CSG325` `15 x 15` `0.8` · `FTG256` `17 x 17` `1.0` · `SBG484` `19 x 19` `0.8` ·
`FGG484` `23 x 23` `1.0` · `FBG484` `23 x 23` `1.0` · `FGG676` `27 x 27` `1.0` ·
`FBG676` `27 x 27` `1.0` · `FFG1156` `35 x 35` `1.0`.
XC7A200T is offered only in `SBG484 (4 GTP, 285 I/O)`, `FBG484 (4, 285)`,
`FBG676 (8, 400)`, `FFG1156 (16, 500)`.
Table 5 Note 2: `Devices in FGG484 and FBG484 are footprint compatible.`

### ABSENT from DS180 (absence is a finding)

- **No QFP/LQFP/TQFP package exists anywhere in Spartan-7 or Artix-7.** Every package in
  Tables 3 and 5 is a BGA (CPGA/CSGA/FTGB/FGGA/CPG/CSG/FTG/SBG/FGG/FBG/FFG). The
  hand-solderable-prototype win we hoped for **does not exist in 7-series**. See §7.5 for the
  partial consolation (1.0 mm pitch).
- **No LUT column.** DS180 gives Slices only.
- **No Zynq-7000 rows.** DS180 covers Spartan-7/Artix-7/Kintex-7/Virtex-7 only; the XC7Z020
  baseline still comes from DS190 (§3.1).
- **No prices, no Fmax, no lifecycle status.**

## 7.2 Prices observed 30/08/2026 (Spartan-7 / Artix-7)

Speed/temperature grade is held at **-1 commercial** wherever possible, to match the baseline
`XC7Z020-1CLG400C`. Where an LCSC line is a different grade it is labelled.

| Part | qty 1 | qty 10 | qty 25/100 | authorized stock | LCSC direct |
|---|---|---|---|---|---|
| **XC7Z020-1CLG400C** *(baseline, §3.5)* | $128.63 Newark · $131.25 Mouser · $131.25 DigiKey | $123.48 Newark | $120.91 @25 Newark | DK 0, Mou 0, New 0 | **$81.7481** (1+), $78.1204 (30+), stock 170 |
| **XC7S50-1FGGA484C** | **$80.97** Newark · $82.62 DigiKey · $102.70 Mouser | $85.90 Mouser | $82.62 @25 Mouser | DK 0, Mou 0, New 0 | **$52.33** (1+), **Out of Stock** |
| **XC7S75-1FGGA484C** | **$113.34** Newark · $115.65 DigiKey · $138.87 Mouser | $117.66 Mouser | not found | DK 0, Mou 0, New 0 | not found |
| **XC7S100-1FGGA484C** | **$152.43** Newark · $155.54 DigiKey · $184.51 Mouser | $146.33 Newark · $157.08 Mouser | $143.28 @25 Newark | DK 0, Mou 0, New 0 | not found |
| **XC7A35T-1CPG236C** | **$55.92** Newark · $57.06 DigiKey · $57.06 Mouser | $53.68 Newark | $52.56 @25 · **$50.33 @100** Newark · $56.69 @100 Mouser | **New 32, Mou 22, DK 72** | not found |
| **XC7A50T-1CSG324C** | **$92.89** Newark · $94.79 DigiKey · $117.83 Mouser | $98.55 Mouser | $94.80 @25 Mouser | DK 0, Mou 0, New 0 | **$29.77** (1+), $28.34 (30+), **stock 5** |
| **XC7A100T-1FGG484C** | **$161.80** Newark · $181.61 DigiKey · $229.25 Mouser | $155.33 Newark · $196.05 Mouser | $152.09 @25 Newark | DK 0, Mou 0, New 0 | **$51.50** (1+), stock 57 |
| **XC7A100T-2FGG484I** *(-2 ind.)* | $235.44 Newark · $240.24 DigiKey · $298.94 Mouser | $226.02 Newark · $257.08 Mouser | $221.31 @25 Newark | DK 0, Mou 0, New 0 | **$70.20** (1+), $65.77 (3+), $63.12 (30+), **stock 657** |
| **XC7A200T-1FBG484C** | **$270.47** Newark · $275.99 DigiKey · $275.99 Mouser | $259.65 Newark | $254.24 @25 Newark | **DK 85 Tray**, Mou 0, New 0 | not found |
| **XC7A200T-2FBG484I** *(-2 ind.)* | $357.36 Newark · $364.65 DigiKey · $446.78 Mouser | $343.07 Newark · $386.49 Mouser | $335.92 @25 Newark | DK 0, **Mou 5**, New 0 | **$120.3205** (1+), $114.5988 (30+), **stock 650** |

**Qty-100 is not found for any Xilinx part above except XC7A35T-1CPG236C** ($50.33 Newark,
$56.69 Mouser) — Newark's break tables stop at 25 for the rest.

**Broker rows (NOT list price; recorded for the counterfeit/grey-market tell only).**
Win Source XC7S100-1FGGA484C `1x $58.65 … 8x $39.10`; IC Components XC7S75 `1x $27.07`
and XC7S50 `1x $20.69`; IC Components XC7A200T-2FBG484I `1x $49.91`; Unikey XC7A200T-2FBG484I
`300 @ $110.495`; Win Source XC7A200T-1FBG484C `1x $147.73`; Unikey XC7A100T-2FGG484I
`1x $62.92`; IC Components XC7A100T-1FGG484C `1x $30.24`. **Component Search** listed
XC7S50-1FGGA484C at `1x $0.01` — an obviously bogus line, quoted here only as evidence that
aggregator broker rows cannot be trusted at all.
`ICHOME Technology`, `Sierra IC`, `GlobalTek Components`, `Perfect Parts`, `Time Way`,
`ICMILES`, `Sense Electronic`, `Unicom`, `Futuretech` were **RFQ-only or OEM/CM-quote-only**
on most parts — no usable price.

## 7.3 Ranking 1 — LUT6 per dollar (PRIMARY)

`LUT6 = Slices × 4` (our derivation from DS180 Note 1). `LUT6/$` = LUT6 ÷ qty-1 price.
**Authorized column uses the lowest authorized qty-1 price from §7.2.**

| Rank | Part | LUT6 | authorized $1 | **LUT6/$** | vs Z-7020 |
|---|---|---|---|---|---|
| **1** | **XC7A200T-1FBG484C** | **134,600** | $270.47 | **498** | **+20%** |
| 2 | XC7S75-1FGGA484C | 48,000 | $113.34 | 424 | +2% |
| 3 | XC7S100-1FGGA484C | 64,000 | $152.43 | 420 | +1% |
| — | **XC7Z020-1CLG400C (baseline)** | **53,200** | **$128.63** | **414** | — |
| 4 | XC7S50-1FGGA484C | 32,600 | $80.97 | 403 | −3% |
| 5 | XC7A100T-1FGG484C | 63,400 | $161.80 | 392 | −5% |
| 6 | XC7A200T-2FBG484I | 134,600 | $357.36 | 377 | −9% |
| 7 | XC7A35T-1CPG236C | 20,800 | $55.92 | 372 | −10% |
| 8 | XC7A50T-1CSG324C | 32,600 | $92.89 | 351 | −15% |
| 9 | XC7A100T-2FGG484I | 63,400 | $235.44 | 269 | −35% |

**LCSC channel (direct-fetched pages), same metric:**

| Rank | Part | LUT6 | LCSC $1 | **LUT6/$** | LCSC stock |
|---|---|---|---|---|---|
| **1** | **XC7A100T-1FGG484C** | 63,400 | $51.50 | **1,231** | 57 |
| 2 | XC7A200T-2FBG484I | **134,600** | $120.3205 | **1,119** | **650** |
| 3 | XC7A50T-1CSG324C | 32,600 | $29.77 | 1,095 | 5 |
| 4 | XC7A100T-2FGG484I | 63,400 | $70.20 | 903 | 657 |
| — | **XC7Z020-1CLG400C (baseline)** | 53,200 | $81.7481 | **651** | 170 |
| 5 | XC7S50-1FGGA484C | 32,600 | $52.33 | 623 | **0 (out of stock)** |

## 7.4 Ranking 2 — block RAM Mb per dollar (SECONDARY)

`BRAM Mb = DS180 "Max (Kb)" ÷ 1024`. Baseline uses DS190's `140` 36 Kb blocks →
`140 × 36 = 5,040 Kb = 4.92 Mb` (our arithmetic; AMD prints `4.9 Mb`).

| Rank | Part | BRAM Kb | BRAM Mb | authorized $1 | **Mb/$** | $/Mb |
|---|---|---|---|---|---|---|
| **1** | **XC7A200T-1FBG484C** | 13,140 | **12.83** | $270.47 | **0.0474** | $21.08 |
| — | **XC7Z020-1CLG400C (baseline)** | 5,040 | 4.92 | $128.63 | **0.0383** | $26.13 |
| 2 | XC7A200T-2FBG484I | 13,140 | 12.83 | $357.36 | 0.0359 | $27.85 |
| 3 | XC7S50-1FGGA484C | 2,700 | 2.64 | $80.97 | 0.0326 | $30.71 |
| 4 | XC7A35T-1CPG236C | 1,800 | 1.76 | $55.92 | 0.0314 | $31.82 |
| 5 | XC7A100T-1FGG484C | 4,860 | 4.75 | $161.80 | 0.0293 | $34.10 |
| 6 | XC7A50T-1CSG324C | 2,700 | 2.64 | $92.89 | 0.0284 | $35.22 |
| 7 | XC7S75-1FGGA484C | 3,240 | 3.16 | $113.34 | 0.0279 | $35.81 |
| 8 | XC7S100-1FGGA484C | 4,320 | 4.22 | $152.43 | 0.0277 | $36.13 |
| 9 | XC7A100T-2FGG484I | 4,860 | 4.75 | $235.44 | 0.0202 | $49.62 |

**LCSC channel:** XC7A200T-2FBG484I **0.1066 Mb/$** ($9.38/Mb) > XC7A100T-1FGG484C 0.0922 >
XC7A50T-1CSG324C 0.0886 > XC7A100T-2FGG484I 0.0677 > **Z-7020 0.0602** > XC7S50 0.0504.

**The two rankings AGREE on the winner: XC7A200T is #1 on both LUT6/$ and BRAM Mb/$, on both
the authorized and the LCSC channel.** They disagree in the middle: the Spartan-7 parts place
2nd/3rd on LUT6/$ but 7th/8th on BRAM/$ — Spartan-7 is logic-rich and BRAM-poor relative to
the BRAM-heavy Zynq. The Z-7020 beats **every** Spartan-7 and every sub-200T Artix-7 on
BRAM/$; it loses to only one part in the whole 7-series lineup.

## 7.5 Does it fit? Utilization of our real designs

LUT6 available vs the two anchor designs (our arithmetic):

| Part | LUT6 | 50,527 LUT (CICIOT46M 500n×34b) | 33,086 LUT | 1,939 LUT (UNSW 92n) |
|---|---|---|---|---|
| XC7A35T | 20,800 | **does not fit** | **does not fit** | 9.3% |
| XC7S50 / XC7A50T | 32,600 | **does not fit** | **does not fit (short by 486 LUT)** | 5.9% |
| XC7A75T | 47,200 | **does not fit** | 70.1% | 4.1% |
| XC7S75 | 48,000 | **does not fit** | 68.9% | 4.0% |
| **XC7Z020 (current)** | **53,200** | **95.0%** | 62.2% | 3.6% |
| XC7S100 | 64,000 | 78.9% | 51.7% | 3.0% |
| XC7A100T | 63,400 | 79.7% | 52.2% | 3.1% |
| **XC7A200T** | **134,600** | **37.5%** | 24.6% | 1.4% |

Two decisions fall straight out of this table:

- **Going UP:** only `XC7S100`, `XC7A100T` and `XC7A200T` give real headroom over the 50,527
  LUT design. `XC7S100`/`XC7A100T` buy ~20% headroom (95% → 79%) at *more* money than the
  Z-7020. `XC7A200T` is the only part that changes the situation qualitatively (95% → 37.5%).
- **Going DOWN:** the 1,939-LUT UNSW 92n design fits `XC7A35T-1CPG236C` at **9.3%**, for
  **$55.92 qty 1 / $50.33 qty 100** — **43% of the Z-7020's price** — and it is the **only
  part in this entire survey with authorized stock on the shelf** (Newark 32, Mouser 22,
  DigiKey 72, all observed 30/08/2026), whereas the Z-7020 shows **0 at all three**. If the
  small-genome IDS variants are the deploy target, this is the strongest result in the section.

## 7.6 Packages and hand-solderability

**Finding: no Spartan-7 or Artix-7 part is offered in any QFP package.** DS180 Tables 3 and 5
list BGA only. Bench prototyping on a bare die means BGA reflow regardless of which 7-series
part is chosen — the Z-7020's `CLG400 17×17 mm 0.8 mm` is not unusually hostile.

The one real (modest) improvement is **ball pitch**: several candidate packages are **1.0 mm**
pitch versus the Z-7020's 0.8 mm, which is a meaningful step for hot-plate/stencil rework:

| Part | Package used for the price above | Size | Pitch | vs Z-7020 (0.8 mm) |
|---|---|---|---|---|
| XC7A35T | `CPG236` | 10 × 10 mm | **0.5 mm** | **worse** |
| XC7A50T | `CSG324` | 15 × 15 mm | 0.8 mm | same |
| XC7S50 | `FGGA484` | 23 × 23 mm | **1.0 mm** | better |
| XC7S75 / XC7S100 | `FGGA484` | 23 × 23 mm | **1.0 mm** | better |
| XC7A100T | `FGG484` | 23 × 23 mm | **1.0 mm** | better |
| XC7A200T | `FBG484` | 23 × 23 mm | **1.0 mm** | better |

`XC7A35T` also exists in `FTG256 (17 × 17 mm, 1.0 mm pitch, 170 I/O)` and `CSG324`, and
`XC7S50` in `FTGB196 (15 × 15 mm, 1.0 mm pitch, 100 I/O)` — **FTGB196 is the smallest 1.0 mm
pitch option in the survey** and the most bench-friendly footprint found, if 100 I/O suffices.
**Prices for the FTG256/FTGB256/FTGB196 orderable parts were not fetched — not found.**

## 7.7 Toolchain

`XC7S50 / XC7S75 / XC7S100 / XC7A35T / XC7A50T / XC7A75T / XC7A100T / XC7A200T` are **all
Vivado parts, same 7-series primitives as the Z-7020's PL** (DS190 §3.1: Z-7020's
`7 Series PL Equivalent` is `Artix-7 FPGA`). Moving to any of them is a **part swap + re-place
and route**, not an RTL port. The one architectural difference to plan for is that
Spartan-7 and Artix-7 have **no PS** — the Z-7020's dual Cortex-A9, its AXI interconnect and
whatever host-side glue currently rides on it must be replaced with an external MCU or a soft
core. **We have not measured what our design actually uses the PS for; that is unresolved and
is the main hidden cost of the Spartan-7/Artix-7 move.** *(An open-source flow exists for
7-series via Yosys + nextpnr-xilinx / Project X-Ray, but it is unofficial and was NOT
evaluated in this run.)*

---

## 7.8 SECONDARY: non-Xilinx families — **each of these requires abandoning Vivado and re-porting the RTL**

This is a real cost, not a footnote. Leaving Vivado means: new synthesis/P&R tool, new
primitive library, new timing-constraint syntax, new bitstream/config flow, re-validation of
every synthesis result already banked (all 21 utilization reports become non-comparable), and
loss of the LUT6-based footprint intuition the whole `project_controller_lut_footprint_b30`
line of work is built on. **None of the parts below is recommended on the numbers alone.**

### Primitive warning — the counts below are NOT comparable to LUT6

Every non-Xilinx part here is built on a **4-input LUT**. A LUT6 holds a 64-entry truth table;
a LUT4 holds 16. For our workload — constant-folded RAM-neuron logic, i.e. wide Boolean
functions of address bits — that difference is **directly load-bearing**, not cosmetic. We
therefore **quote raw vendor counts and refuse to convert between them.** Where a vendor
inflates the number, it is called out:

- **Lattice CertusPro-NX** datasheet Note 1, verbatim: `Logic Cells = LUTs × 1.2
  effectiveness.` So `96k` logic cells on LFCPNX-100 is **80,000 LUT4**, not 96,000.
- **AMD** "Logic Cells" are likewise inflated (XC7A200T: `215,360` logic cells but
  `33,650` slices = 134,600 LUT6). **Never rank on a "logic cell" column.**
- **Efinix** T20/T120 datasheets, verbatim: `Logic capacity in equivalent LE counts.`
  "Equivalent" is the vendor's own hedge.

### Verbatim device data

| Family / part | Primitive (verbatim) | Count | LUTRAM / distributed | Block RAM | Toolchain | Open-source flow |
|---|---|---|---|---|---|---|
| **Efinix Trion T120** | `The logic cell comprises a 4-input LUT or a full adder plus a register (flipflop)` | `112,128` LEs | **NONE — the concept does not exist in the datasheet** | `5,407` kbits in `1,056` × 5 Kbit blocks | Efinity | no |
| **Efinix Trion T20** | same | `19,728` LEs | **NONE** | `1044.48` kbits in `204` × 5 Kbit blocks | Efinity | no |
| **Lattice ECP5 LFE5U-85** | `Each slice contains two LUT4s feeding two registers` | `84` K LUTs | `669` Kbits | `3744` Kbits, `208` × 18 Kbit EBR | Diamond/Radiant | **yes — Yosys + nextpnr-ecp5** |
| **Lattice ECP5 LFE5U-45** | same | `44` K LUTs | `351` Kbits | `1944` Kbits, `108` × 18 Kbit EBR | Diamond/Radiant | **yes** |
| **Lattice CertusPro-NX LFCPNX-100** | LUT4 (`Logic Cells = LUTs × 1.2`) | `96k` LC = **80,000 LUT4** | `639` kb | `3,744` kb EBR **+ `3,584` kb LRAM** (`7` × 512 kb) | Radiant | partial (nextpnr-nexus) |
| **Lattice CertusPro-NX LFCPNX-50** | same | `52k` LC = **43,333 LUT4** | `344` kb | `1,728` kb EBR + `2,048` kb LRAM | Radiant | partial |
| **Gowin GW5A-25** | `LUT4` | `23040` | SSRAM `180` Kb | BSRAM `1008` Kb, `56` blocks | Gowin EDA | partial (apicula) |
| **Gowin GW5A-60** | `LUT4` | `59904` | SSRAM `468` Kb | BSRAM `2124` Kb, `118` blocks | Gowin EDA | partial |
| **Gowin GW5A-138** | `LUT4` | `138240` | SSRAM `1080` Kb | BSRAM `6120` Kb, `340` blocks | Gowin EDA | partial |
| **Gowin GW2A-18** | `LUT4s` | `20,736` | SSRAM `40K` bits | BSRAM `828K` bits, `46` blocks | Gowin EDA | partial |
| **Gowin GW2A-55** | `LUT4s` | `54,720` | SSRAM `106K` bits | BSRAM `2,520K` bits, `140` blocks | Gowin EDA | partial |
| **Microchip PolarFire MPF100T** | `4-input look-up table (LUT) with a fractureable D-type flipflop` | `109` K LE | µSRAM `1008` × `64 × 12` blocks | LSRAM `352` × 20 kbit; **Total RAM `7.6` Mbits** | Libero SoC | no |
| **Microchip PolarFire MPF300T** | same | `300` K LE | µSRAM `2772` blocks | LSRAM `952`; **Total RAM `20.6` Mbits** | Libero SoC | no |
| **Intel/Altera MAX 10 10M50** | Logic Element (4-LUT) | `50` K LE | not found in this document | M9K `1,638` Kb | Quartus | no |
| **Intel/Altera Cyclone 10 LP** | — | **NOT VERIFIED — product-table fetch returned an XML error page; no primary figure read** | — | — | Quartus | no |
| **Efinix Titanium Ti60** | — | **NOT VERIFIED — datasheet not fetched in this run; a search snippet claimed 62,000 LE but that was not read from the PDF** | — | — | Efinity | no |

### Secondary prices (30/08/2026) and ratios

| Part | Best authorized qty 1 | qty 10/25/100 | Stock | Count/$ | BRAM Mb/$ |
|---|---|---|---|---|---|
| **Efinix T120F484C4** | **$52.92** DigiKey | Verical $73.79 @10, $70.66 @50 | **DigiKey 1,238** | **2,119 LE/$** | **0.0998** |
| Efinix T20F256C3 | **$13.18** DigiKey | $11.56 @25 DigiKey | **DigiKey 11,744** | 1,497 LE/$ | 0.0774 |
| Lattice LFE5U-85F-6BG381C | **$74.75** DigiKey/Mouser | $65.12 @25, $60.28 @100 DigiKey | DK 0, Mou 0, New 0 | 1,124 LUT4/$ | 0.0489 |
| Lattice LFE5U-45F-6BG381C | **$46.64** DigiKey | $40.70 @25, $37.62 @100 DigiKey; Arrow/Verical $34.01 @1 | **DigiKey 46**, Mouser 5, Arrow 630 | 943 LUT4/$ | 0.0407 |
| Microchip MPF100T-FCG484I | **$153.49** Future Electronics · $170.91 Newark · $174.40 DigiKey/Mouser/Microchip-direct | $149.55 @10, $138.87 @25 Newark; $141.70 @25 DigiKey | **DigiKey 5, Mouser 2**; Microchip direct 0, **lead time 30 weeks** | 638 LE/$ | 0.0445 |
| Gowin GW5A-LV25MG121NC1/I0 | **not found at any authorized distributor** | — | — | — | — |
| Gowin GW5A-LV25MG121NC1/I0 (LCSC) | $46.8761 (1+), $44.5536 (30+) | — | LCSC 459 | 492 LUT4/$ | 0.0210 |
| Lattice LFCPNX-100 / LFCPNX-50 | **not found** (not priced in this run) | — | — | — | — |
| Intel MAX 10 10M50 | **not found** (not priced in this run) | — | — | — | — |
| Efinix Ti60 | **not found** | — | — | — | — |

*(Reference: Z-7020 = 414 LUT6/$ and 0.0383 BRAM Mb/$ at $128.63 authorized.)*

**What the secondary numbers say, with the primitive caveat applied:**

- **Efinix T120 is the only non-Xilinx part that is genuinely dominant on both axes:**
  `112,128` LEs and `5,407` kbits (5.28 Mb) of block RAM — **more BRAM than the Z-7020's
  4.92 Mb** — for **$52.92 at DigiKey with 1,238 in stock**. That is 2.6× the BRAM per dollar
  and, at face value, 5× the "logic count" per dollar. **But** its LEs are LUT4-equivalents
  (vendor's own word), it has **zero distributed RAM of any kind**, and it means Efinity
  instead of Vivado. If the sparse-keys+binary-search representation is adopted (making BRAM
  binding), T120 is the part to argue about; on the current all-combinational representation
  the LUT4-vs-LUT6 gap makes the LE headline untrustworthy.
- **ECP5 is a downgrade on both axes despite being cheap.** LFE5U-85F has `3744` Kbits EBR
  = **3.66 Mb, i.e. 26% LESS block RAM than the Z-7020**, and its 84 K LUT**4**s cannot be
  compared to 53,200 LUT**6**s. Its one genuine advantage is the fully open Yosys/nextpnr-ecp5
  flow.
- **Gowin is the worst value in the survey** (492 LUT4/$, 0.0210 Mb/$ — below the Z-7020 on
  both) and had **no authorized distributor listing at all**; the only price is LCSC. The
  large `GW5A-138` (138,240 LUT4, 6,120 Kb BSRAM) would change that, but **no price was found
  for it — not found.**
- **PolarFire MPF100T** has the most on-chip RAM per part of the sub-$200 group (`7.6` Mbits)
  but at 0.0445 Mb/$ it barely beats the Z-7020's 0.0383 and loses badly to XC7A200T's 0.0474
  — while costing a full Libero re-port and carrying a **30-week factory lead time**.

---

## 7.9 Verdict

1. **The "Zynq tax" hypothesis is REFUTED at authorized pricing.** The premise was that the
   Z-7020's price includes a hard Cortex-A9 we may not need, so a pure FPGA should give far
   more LUTs per dollar. It does not. At authorized qty 1 the Z-7020 delivers **414 LUT6/$**,
   and the *only* 7-series part that beats it is the XC7A200T (498). Spartan-7 — the pure-FPGA,
   no-PS, cost-optimized family that should have won this outright — lands at 424 / 420 / 403,
   i.e. **within ±3% of the Zynq**. There is no free lunch in dropping the PS.
2. **Strictly, the literal question has no good answer on the authorized channel.** Nothing
   offers more than 53,200 LUT6 for ≤ $128.63. XC7S100 (+20% LUTs) costs +19%; XC7A100T
   (+19% LUTs) costs +26%; XC7A200T (+153% LUTs) costs +110%.
3. **XC7A200T is the answer if the budget can move**, and it wins *both* rankings on *both*
   channels: 134,600 LUT6 (2.53×) and 12.83 Mb BRAM (2.61×) for 2.10× the authorized price —
   and it takes the 50,527-LUT design from 95.0% to **37.5%** utilization. **On LCSC it is
   $120.32 with 650 in stock, i.e. below the Z-7020's own *authorized* price of $128.63**,
   though above the Z-7020's LCSC price of $81.75.
4. **The cheapest real win may be downward, not upward.** `XC7A35T-1CPG236C` at $55.92 qty 1 /
   $50.33 qty 100 fits the 1,939-LUT UNSW 92n design at 9.3%, and it is the only part in the
   survey **with authorized stock actually on the shelf**. Its `CPG236` package is 0.5 mm
   pitch (worse than the Z-7020 for bench work); `FTG256` at 1.0 mm exists but was not priced.
5. **No hand-solderable option exists.** 7-series is BGA-only, full stop. The achievable win
   is 0.8 mm → 1.0 mm pitch (FGG484/FBG484/FTG256/FTGB196), not QFP.
6. **Every non-Xilinx candidate costs an RTL port**, and only **Efinix T120** is compelling
   enough on raw numbers to justify even discussing it — and only under the BRAM-binding
   representation, since its LUT4-equivalent "LE" count is not comparable to LUT6.

## 7.10 NOT verified in §7 — do not cite these

- **Any price read off a Digi-Key / Mouser / Newark / Arrow / Verical / Future page directly.**
  All still 403/ECONNRESET on 30/08/2026; every such figure in §7.2 and §7.8 is
  oemstrade-aggregator-reported. Only the LCSC rows were fetched from the vendor's own page.
- **Search-snippet prices are actively wrong.** A web-search snippet claimed LCSC
  `XC7A100T-2FGG484I` at `$27.2888` and `XC7S50-1FGGA484C` at `$23.1877`. The **fetched LCSC
  pages** say **$70.20** and **$52.33**. The snippets were discarded. Never quote an LCSC price
  from a search snippet.
- **Qty-100 for every Xilinx part except XC7A35T-1CPG236C.** Newark's tables stop at 25.
- **LCSC listings for XC7S75, XC7S100, XC7A200T-1FBG484C, XC7A35T** — not found.
- **Prices for the FTG256 / FTGB196 orderable part numbers** (the small 1.0 mm-pitch packages)
  — not fetched. Not found.
- **XC7Z020 distributed-RAM total** — still unpublished by AMD (unchanged from §6). Note that
  DS180 *does* publish it for Spartan-7/Artix-7, which is why the §7.1 tables have the column
  and the baseline does not. Irrelevant to the ranking now that distributed RAM is known unused.
- **Intel Cyclone 10 LP** — the product-table URL returned an XML error page. No primary
  figure was read. Everything about Cyclone 10 LP is **not found**.
- **Efinix Titanium Ti60** — datasheet not fetched. The "62,000 LE" figure appeared only in a
  search snippet and is **not verified**.
- **Prices for Lattice CertusPro-NX, Intel MAX 10, Gowin GW5A-138, Gowin GW2A-55** — not found.
- **Gowin GW5A-25 authorized distribution** — no authorized distributor listing was found at
  all; the LCSC line is the only price.
- **Lifecycle status** for every part in §7 — no distributor status field could be read. No
  EOL/NRND notice was found for any of them, but "Active" is inferred, not read.
- **PL Fmax / achievable clock** for any part in §7 — design-dependent, not a datasheet number.
- **Open-source-flow claims** (Yosys/nextpnr-xilinx, nextpnr-ecp5, nextpnr-nexus, apicula) —
  stated from general knowledge, **NOT verified against a project page in this run**. Treat the
  "Open-source flow" column of §7.8 as unverified.
- **What our design actually uses the Z-7020 PS for** — never measured. This is the main
  unpriced cost of any move to a PS-less part (all of Spartan-7 and Artix-7).
- **DS180 version drift:** the tables in §7.1 are from **v2.5 (August 1, 2017)**. A later
  **v2.6.1 (September 8, 2020)** exists; Mouser's copy refused automated fetch. Device
  resource counts are not expected to change between revisions but this was **not confirmed**.

## 7.11 Sources added in §7

- AMD/Xilinx, *7 Series FPGAs Data Sheet: Overview*, DS180 (v2.5), August 1, 2017 —
  <https://www.farnell.com/datasheets/2553941.pdf>
  (local: `papers/xilinx2017_ds180_7series_overview_v25.pdf`)
- Lattice, *ECP5 Family Data Sheet*, DS1044 v1.2, August 2014 (preliminary) —
  <https://datasheet.octopart.com/LFE5U-85F-8BG381C-Lattice-Semiconductor-datasheet-42411106.pdf>
  (local: `papers/lattice2014_ds1044_ecp5_family.pdf`)
- Lattice, *CertusPro-NX Family Data Sheet*, FPGA-DS-02086-2.2, January 2025 —
  <https://mm.digikey.com/Volume0/opasdata/d220001/medias/docus/6971/CertusPro-NX%20Family%20Datasheet.pdf>
  (local: `papers/lattice2025_certuspro_nx_family_ds.pdf`)
- Efinix, *T120 Data Sheet*, DST120-v4.0, November 2025 —
  <https://www.efinixinc.com/docs/trion120-ds-v4.0.pdf>
  (local: `papers/efinix2025_dst120_trion_t120.pdf`)
- Efinix, *T20 Data Sheet*, DST20-v1.0, May 2019 —
  <https://www.efinixinc.com/docs/trion20-ds-v1.0.pdf>
  (local: `papers/efinix2019_dst20_trion_t20.pdf`)
- Gowin, *GW5A series of FPGA Products Data Sheet*, DS1103-1.0.9E —
  <https://cdn.gowinsemi.com.cn/DS1103E.pdf>
  (local: `papers/gowin2025_ds1103_gw5a_family.pdf`)
- Gowin, *GW2A series of FPGA Products Data Sheet*, DS102-2.7.8E, 07/10/2026 —
  <https://cdn.gowinsemi.com.cn/DS102E.pdf>
  (local: `papers/gowin2026_ds102_gw2a_family.pdf`)
- Microchip, *PolarFire FPGA Product Overview*, DS60001657R, © 2017–2025 —
  <https://ww1.microchip.com/downloads/aemDocuments/documents/FPGA/ProductDocuments/ProductBrief/PolarFire-FPGA-Product-Overview-60001657.pdf>
  (local: `papers/microchip2025_polarfire_product_overview.pdf`)
- Intel, *Intel MAX 10 FPGA Device Overview* —
  <https://pages.hmc.edu/brake/class/e155/fa21/assets/doc/MAX_10_Device_Overview.pdf>
  (local: `papers/intel_max10_device_overview.pdf`)
- oemstrade aggregator, observed 30/08/2026 —
  `/search/` for `xc7s50-1fgga484c` · `xc7s75-1fgga484c` · `xc7s100` · `xc7s100-1fgga484c` ·
  `xc7a35t-1cpg236c` · `xc7a50t-1csg324c` · `xc7a100t-1fgg484c` · `xc7a100t-2fgg484i` ·
  `xc7a200t-1fbg484c` · `xc7a200t-2fbg484i` · `lfe5u-45f-6bg381c` · `lfe5u-85f-6bg381c` ·
  `mpf100t-fcg484i` · `t20f256c3` · `t120f484c4` · `gw5a-lv25`
- LCSC (direct page fetch, 30/08/2026) —
  <https://www.lcsc.com/product-detail/C494680.html> (XC7A200T-2FBG484I) ·
  <https://www.lcsc.com/product-detail/C1521767.html> (XC7A100T-1FGG484C) ·
  <https://www.lcsc.com/product-detail/C410276.html> (XC7A100T-2FGG484I) ·
  <https://www.lcsc.com/product-detail/C1527306.html> (XC7S50-1FGGA484C)
