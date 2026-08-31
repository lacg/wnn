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
