#!/bin/zsh
# CTRL-69 offline pre-check — exact reproduction (CPU-light; no flight, no venv install).
# 1) scratch wheel: apply trace_dagger_cascade.patch to a private worktree of 269e9e02 (ABI 60 source),
#    build with a private CARGO_TARGET_DIR, install to a temp --target (never the venv):
#      git worktree add --detach /private/tmp/ctrl69_wt 269e9e02 && cd /private/tmp/ctrl69_wt && git apply <this dir>/trace_dagger_cascade.patch
#      cd src/wnn/ram/strategies/accelerator && CARGO_TARGET_DIR=/Volumes/20260401-WDBlack-SN850X-2TB/cargo-target-ctrl69 \
#        nice -n 10 maturin build --release -m controller/Cargo.toml -o /private/tmp/ctrl69_wheel
#      python -m pip install --no-deps --target /private/tmp/ctrl69_pkgs /private/tmp/ctrl69_wheel/ram_controller-*.whl
set -e
cd /Users/lacg/wnn
export PYTHONPATH=/Users/lacg/wnn/src
PY=/Volumes/20260401-WDBlack-SN850X-2TB/wnn/venv/bin/python
D=logs/controller/ctrl69
N0=logs/controller/ctrl48/C-S2-d_wposx2_a60_rf_c200_dc_biasin_ms01_P_both_L1_yld_n0_full_s1
YFL=logs/controller/ctrl48/C-S2-d_wposx2_a59_rf_c200_dc_biasin_ms01_P_both_L1_ffF_yld_full_s1
for s in 69000001:fit 69000002:val; do
	nice -n 10 $PY $D/collect.py $N0.out /private/tmp/ctrl69_pkgs /private/tmp/ctrl69_${s#*:}.npz ${s%:*} 100                        # PREREG: teacher-driven
	nice -n 10 $PY $D/collect.py $N0.out /private/tmp/ctrl69_pkgs /private/tmp/ctrl69_n0_${s#*:}.npz ${s%:*} 100 ${N0}_winner.yaml.gz # post-hoc: N0 winner flies
	nice -n 10 $PY $D/collect.py $YFL.out /private/tmp/ctrl69_pkgs /private/tmp/ctrl69_st_${s#*:}.npz ${s%:*} 100 ${YFL}_winner.yaml.gz # post-hoc: Y+F+L winner flies
done
nice -n 10 $PY $D/analyze.py /private/tmp/ctrl69_fit.npz /private/tmp/ctrl69_val.npz > $D/analyze_teacher.out
nice -n 10 $PY $D/analyze.py /private/tmp/ctrl69_fit.npz /private/tmp/ctrl69_val.npz --prev-label > $D/analyze_teacher_prevlabel.out
nice -n 10 $PY $D/analyze.py /private/tmp/ctrl69_n0_fit.npz /private/tmp/ctrl69_n0_val.npz > $D/analyze_student_n0_s1.out
nice -n 10 $PY $D/analyze.py /private/tmp/ctrl69_st_fit.npz /private/tmp/ctrl69_st_val.npz > $D/analyze_student_yfl_s1.out
