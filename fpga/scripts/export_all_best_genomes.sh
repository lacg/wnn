#!/usr/bin/env bash
# Export all best genomes for Vivado synthesis
# Run from project root: bash fpga/scripts/export_all_best_genomes.sh
#
# After export, upload fpga/export/ to EC2 and run synth_all.tcl

set -euo pipefail
cd "$(dirname "$0")/../.."

EXPORT="python fpga/scripts/export_genome.py"

echo "======================================================"
echo " Exporting best genomes for Vivado synthesis"
echo "======================================================"

# 1. UNSW temporal Best F1: flow 1567, GS best_f1, 100n
echo -e "\n[1/8] UNSW temporal Best F1 (100n, 90.52% F1)"
$EXPORT --flow-id 1567 --genome-type best_f1 --phase grid_search \
	--output fpga/export/unsw_temporal_best_f1

# 2. UNSW temporal Best FPR: flow 1489, GA best_f1, 200n
echo -e "\n[2/8] UNSW temporal Best FPR (200n, 71.80% F1 / 0% FPR)"
$EXPORT --flow-id 1489 --genome-type best_f1 --phase ga_neurons \
	--output fpga/export/unsw_temporal_best_fpr

# 3. CICIDS Best F1: flow 936, GA best_ce, 94n
echo -e "\n[3/8] CICIDS Best F1 (94n, 99.55% F1)"
$EXPORT --flow-id 936 --genome-type best_ce --phase ga_neurons \
	--output fpga/export/cicids_best_f1

# 4. CIC-IoT 1.3M Best F1: flow 789, GA best_f1, 198n
echo -e "\n[4/8] CIC-IoT Best F1 (198n, 83.04% F1)"
$EXPORT --flow-id 789 --genome-type best_f1 --phase ga_neurons \
	--output fpga/export/ciciot_best_f1

# 5. CIC-IoT 1.3M Best F1 FPR<14%: flow 796, GA best_ce, 374n
echo -e "\n[5/8] CIC-IoT Best F1 FPR<14% (374n, 82.76% F1)"
$EXPORT --flow-id 796 --genome-type best_ce --phase ga_neurons \
	--output fpga/export/ciciot_best_f1_fpr14

# 6. CIC-IoT 1.3M Best F1 FPR<10%: flow 794, GA best_acc, 245n
echo -e "\n[6/8] CIC-IoT Best F1 FPR<10% (245n, 82.45% F1)"
$EXPORT --flow-id 794 --genome-type best_acc --phase ga_neurons \
	--output fpga/export/ciciot_best_f1_fpr10

# 7. CIC-IoT 1.3M Best F1 FPR<5%: flow 752, GA best_ce, 299n
echo -e "\n[7/8] CIC-IoT Best F1 FPR<5% (299n, 81.14% F1)"
$EXPORT --flow-id 752 --genome-type best_ce --phase ga_neurons \
	--output fpga/export/ciciot_best_f1_fpr5

# 8. CIC-IoT 1.3M Best FPR: flow 837, GS best_fpr, 5n
echo -e "\n[8/8] CIC-IoT Best FPR (5n, 70.34% F1 / 0.03% FPR)"
$EXPORT --flow-id 837 --genome-type best_fpr --phase grid_search \
	--output fpga/export/ciciot_best_fpr

echo -e "\n======================================================"
echo " All 8 genomes exported to fpga/export/"
echo " Next: generate RTL, upload to EC2, run Vivado"
echo "======================================================"
