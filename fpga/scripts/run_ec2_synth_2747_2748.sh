#!/usr/bin/env bash
# Orchestrate EC2 Vivado synth for the 2747+2748 cohort: start instance,
# upload bundles, run synth, pull results, stop instance.
#
# CRITICAL: traps stop the instance on EVERY exit path (including failure /
# interrupt) so we never leave the $0.34/hr c5.2xlarge running by accident.
# (per memory: reference_aws_vivado_ec2 — instance was found running 24/7 for
# 24 days once, ~$196 wasted; never again.)

set -euo pipefail

AWS_PROFILE=laray
AWS_REGION=us-east-1
INSTANCE_ID=i-08d43449d89148737
SSH_KEY=$HOME/.ssh/vivado-synth.pem
SSH_USER=ubuntu
REMOTE_BASE=/home/ubuntu/wnn_fpga
LOCAL_REPO=/Users/lacg/wnn
LOCAL_RESULTS=$LOCAL_REPO/fpga/synth_results_collected/results

BUNDLES=(
    flow_2747_best_fpr
    flow_2747_best_f1_logic
    flow_2747_best_fitness_logic
    flow_2748_best_fitness_logic
)

log() { echo "[$(date '+%H:%M:%S')] $*"; }

stop_instance() {
    log "STOPPING EC2 instance ${INSTANCE_ID} (always-stop trap)"
    aws ec2 stop-instances --instance-ids "$INSTANCE_ID" \
        --profile "$AWS_PROFILE" --region "$AWS_REGION" \
        --query "StoppingInstances[0].CurrentState.Name" --output text 2>&1 \
        | sed 's/^/  /'
}
trap stop_instance EXIT INT TERM

# Ensure key has the right perms (SSH rejects loose perms).
chmod 400 "$SSH_KEY" 2>/dev/null || true

log "STARTING EC2 instance ${INSTANCE_ID}"
aws ec2 start-instances --instance-ids "$INSTANCE_ID" \
    --profile "$AWS_PROFILE" --region "$AWS_REGION" \
    --query "StartingInstances[0].CurrentState.Name" --output text > /dev/null

log "Waiting for instance to reach 'running'..."
aws ec2 wait instance-running --instance-ids "$INSTANCE_ID" \
    --profile "$AWS_PROFILE" --region "$AWS_REGION"

IP=$(aws ec2 describe-instances --instance-ids "$INSTANCE_ID" \
        --profile "$AWS_PROFILE" --region "$AWS_REGION" \
        --query "Reservations[0].Instances[0].PublicIpAddress" --output text)
log "Instance running at ${IP}"

# Give sshd a few seconds to come up post-boot.
for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20; do
    if ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 -i "$SSH_KEY" \
            "${SSH_USER}@${IP}" "echo ok" >/dev/null 2>&1; then
        log "SSH up after ${i}s"
        break
    fi
    sleep 1
done

# Prep remote dirs (idempotent).
log "Preparing remote layout under ${REMOTE_BASE}/"
ssh -o StrictHostKeyChecking=no -i "$SSH_KEY" "${SSH_USER}@${IP}" \
    "mkdir -p ${REMOTE_BASE}/export ${REMOTE_BASE}/rtl"

# Upload the shared sparse-neuron RTL once (for the standard small-genome synth).
log "Uploading shared RTL"
scp -o StrictHostKeyChecking=no -i "$SSH_KEY" -q \
    ${LOCAL_REPO}/fpga/rtl/wnn_neuron.sv \
    ${LOCAL_REPO}/fpga/rtl/wnn_neuron_dense.sv \
    ${LOCAL_REPO}/fpga/rtl/wnn_classifier.sv \
    "${SSH_USER}@${IP}:${REMOTE_BASE}/rtl/"

# Upload each genome bundle.
for b in "${BUNDLES[@]}"; do
    log "Uploading bundle ${b}"
    scp -o StrictHostKeyChecking=no -i "$SSH_KEY" -q -r \
        ${LOCAL_REPO}/fpga/export/${b} \
        "${SSH_USER}@${IP}:${REMOTE_BASE}/export/"
done

# Upload the batch synth TCL.
log "Uploading synth TCL"
scp -o StrictHostKeyChecking=no -i "$SSH_KEY" -q \
    ${LOCAL_REPO}/fpga/scripts/synth_2747_2748_cohort.tcl \
    "${SSH_USER}@${IP}:${REMOTE_BASE}/"

# Run Vivado. The vivado binary is on PATH on the instance per existing scripts.
log "Running Vivado batch synth (this is the long step)"
ssh -o StrictHostKeyChecking=no -i "$SSH_KEY" "${SSH_USER}@${IP}" \
    "cd ${REMOTE_BASE} && vivado -mode batch -source synth_2747_2748_cohort.tcl 2>&1 | tail -80"

# Pull results back. Each bundle now has utilization.rpt / timing.rpt /
# power.rpt / summary.txt; copy the whole bundle to preserve any logs.
mkdir -p "${LOCAL_RESULTS}"
for b in "${BUNDLES[@]}"; do
    log "Downloading results for ${b}"
    mkdir -p "${LOCAL_RESULTS}/${b}"
    scp -o StrictHostKeyChecking=no -i "$SSH_KEY" -q \
        "${SSH_USER}@${IP}:${REMOTE_BASE}/export/${b}/{utilization.rpt,timing.rpt,power.rpt,summary.txt}" \
        "${LOCAL_RESULTS}/${b}/" 2>&1 | head -5 || true
done

log "Synth complete. Results in ${LOCAL_RESULTS}/"

# Trap will stop the instance now.
