//! Flow lifecycle queries: stop/delete/heartbeat/stale-requeue/restart (split from db/mod.rs `queries`).

use super::*;
use super::queries::*;

pub async fn delete_flow(pool: &DbPool, id: i64) -> Result<bool> {
    // Stop any running process first
    stop_flow_process(pool, id).await?;

    // Delete all associated data (experiments, phases, iterations, checkpoints, junction table)
    delete_flow_data(pool, id).await?;

    // Delete the flow itself
    let result = sqlx::query("DELETE FROM flows WHERE id = ?")
        .bind(id)
        .execute(pool)
        .await?;

    Ok(result.rows_affected() > 0)
}

/// Stop a running flow process by sending SIGTERM.
/// This is reusable by delete_flow, update_flow_for_restart, etc.
///
/// Always updates status to 'cancelled', even if PID is missing.
/// The worker will check flow status and stop gracefully.
pub async fn stop_flow_process(pool: &DbPool, flow_id: i64) -> Result<()> {
    // Get the flow's PID
    let pid: Option<i64> = sqlx::query_scalar(
        "SELECT pid FROM flows WHERE id = ?"
    )
    .bind(flow_id)
    .fetch_optional(pool)
    .await?
    .flatten();

    // Try to send SIGTERM if we have a PID
    if let Some(pid) = pid {
        #[cfg(unix)]
        {
            // Send SIGTERM to gracefully stop the process
            let result = unsafe { libc::kill(pid as i32, libc::SIGTERM) };
            if result == 0 {
                tracing::info!("Sent SIGTERM to flow {} (PID {})", flow_id, pid);
            } else {
                tracing::warn!("Failed to send SIGTERM to flow {} (PID {})", flow_id, pid);
            }
        }
    } else {
        tracing::warn!("No PID registered for flow {}, marking as cancelled (worker will check status)", flow_id);
    }

    // Clear the PID regardless of status, but flip to cancelled ONLY from
    // running/queued: a flow that reached a terminal status between the
    // caller's check and this update must keep it (the old unconditional
    // UPDATE could flip completed → cancelled — TOCTOU, wipe-bug family).
    // Single transaction so pid-clear + cancel + cascade land together.
    let mut tx = pool.begin().await?;

    sqlx::query("UPDATE flows SET pid = NULL WHERE id = ?")
        .bind(flow_id)
        .execute(&mut *tx)
        .await?;

    let cancelled = sqlx::query(
        "UPDATE flows SET status = 'cancelled', completed_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now') \
         WHERE id = ? AND status IN ('running', 'queued', 'pending', 'paused')"
    )
    .bind(flow_id)
    .execute(&mut *tx)
    .await?;

    // Also cancel all running experiments linked to this flow
    if cancelled.rows_affected() > 0 {
        sqlx::query(
            "UPDATE experiments SET status = 'cancelled', ended_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now') WHERE flow_id = ? AND status = 'running'"
        )
        .bind(flow_id)
        .execute(&mut *tx)
        .await?;
    }

    tx.commit().await?;
    Ok(())
}

/// Delete all data associated with a flow (experiments, iterations, checkpoints)
/// This is reused by both delete_flow and update_flow_for_restart
///
/// Uses flow_id foreign keys for clean cascade deletion:
/// experiments.flow_id -> iterations -> genome_evaluations
async fn delete_flow_data(pool: &DbPool, flow_id: i64) -> Result<()> {
    // Clear seed_checkpoint_id from flow FIRST (to remove FK dependency)
    sqlx::query("UPDATE flows SET seed_checkpoint_id = NULL WHERE id = ?")
        .bind(flow_id)
        .execute(pool)
        .await?;

    // Get all experiments for this flow
    let exp_ids: Vec<i64> = sqlx::query_scalar(
        "SELECT id FROM experiments WHERE flow_id = ?"
    )
    .bind(flow_id)
    .fetch_all(pool)
    .await?;

    for exp_id in &exp_ids {
        delete_experiment_data(pool, *exp_id).await?;
    }

    // Delete experiments by flow_id
    sqlx::query("DELETE FROM experiments WHERE flow_id = ?")
        .bind(flow_id)
        .execute(pool)
        .await?;

    Ok(())
}

/// Clear display data for an experiment (iterations, genome_evaluations, genomes)
/// but KEEP checkpoints so resume can seed from them, and KEEP
/// validation_summaries: they double as the cross-flow validation cache
/// (get_cached_validation) — deleting them on a restart-resume threw away
/// hours of full-dataset validations; they're upserted, so re-runs
/// overwrite stale rows naturally.
///
/// Runs in a single transaction; best_genomes rows referencing this
/// experiment's genomes are removed FIRST (foreign_keys=ON — deleting
/// genomes with live best_genomes refs aborted the whole restart with an
/// FK violation mid-mutation).
async fn clear_experiment_display_data(pool: &DbPool, exp_id: i64) -> Result<()> {
    let mut tx = pool.begin().await?;

    // Delete health checks for iterations
    sqlx::query(
        "DELETE FROM health_checks WHERE iteration_id IN (SELECT id FROM iterations WHERE experiment_id = ?)"
    )
    .bind(exp_id)
    .execute(&mut *tx)
    .await?;

    // Delete genome evaluations for iterations
    sqlx::query(
        "DELETE FROM genome_evaluations WHERE iteration_id IN (SELECT id FROM iterations WHERE experiment_id = ?)"
    )
    .bind(exp_id)
    .execute(&mut *tx)
    .await?;

    // Delete iterations
    sqlx::query("DELETE FROM iterations WHERE experiment_id = ?")
        .bind(exp_id)
        .execute(&mut *tx)
        .await?;

    // Delete best_genomes referencing this experiment's genomes (FK)
    sqlx::query(
        "DELETE FROM best_genomes WHERE genome_id IN (SELECT id FROM genomes WHERE experiment_id = ?)"
    )
    .bind(exp_id)
    .execute(&mut *tx)
    .await?;

    // Delete genome_evaluations that reference genomes
    sqlx::query(
        "DELETE FROM genome_evaluations WHERE genome_id IN (SELECT id FROM genomes WHERE experiment_id = ?)"
    )
    .bind(exp_id)
    .execute(&mut *tx)
    .await?;

    // Delete genomes
    sqlx::query("DELETE FROM genomes WHERE experiment_id = ?")
        .bind(exp_id)
        .execute(&mut *tx)
        .await?;

    tx.commit().await?;
    Ok(())
}

/// Delete all data for an experiment (iterations, genome_evaluations, genomes, checkpoints, validation_summaries, gating_runs)
pub(crate) async fn delete_experiment_data(pool: &DbPool, exp_id: i64) -> Result<()> {
    // Delete gating runs for this experiment
    sqlx::query("DELETE FROM gating_runs WHERE experiment_id = ?")
        .bind(exp_id)
        .execute(pool)
        .await?;

    // Delete validation summaries for this experiment
    sqlx::query("DELETE FROM validation_summaries WHERE experiment_id = ?")
        .bind(exp_id)
        .execute(pool)
        .await?;

    // Delete health checks for iterations of this experiment
    sqlx::query(
        "DELETE FROM health_checks WHERE iteration_id IN (SELECT id FROM iterations WHERE experiment_id = ?)"
    )
    .bind(exp_id)
    .execute(pool)
    .await?;

    // Delete genome evaluations for iterations of this experiment
    sqlx::query(
        "DELETE FROM genome_evaluations WHERE iteration_id IN (SELECT id FROM iterations WHERE experiment_id = ?)"
    )
    .bind(exp_id)
    .execute(pool)
    .await?;

    // Delete iterations for this experiment
    sqlx::query("DELETE FROM iterations WHERE experiment_id = ?")
        .bind(exp_id)
        .execute(pool)
        .await?;

    // Delete best_genomes that reference genomes from this experiment
    sqlx::query(
        "DELETE FROM best_genomes WHERE genome_id IN (SELECT id FROM genomes WHERE experiment_id = ?)"
    )
    .bind(exp_id)
    .execute(pool)
    .await?;

    // Delete genome_evaluations that reference genomes from this experiment
    sqlx::query(
        "DELETE FROM genome_evaluations WHERE genome_id IN (SELECT id FROM genomes WHERE experiment_id = ?)"
    )
    .bind(exp_id)
    .execute(pool)
    .await?;

    // Delete genomes
    sqlx::query("DELETE FROM genomes WHERE experiment_id = ?")
        .bind(exp_id)
        .execute(pool)
        .await?;

    // Get checkpoint file paths before deleting records
    let checkpoint_paths: Vec<String> = sqlx::query_scalar(
        "SELECT file_path FROM checkpoints WHERE experiment_id = ?"
    )
    .bind(exp_id)
    .fetch_all(pool)
    .await?;

    // Delete checkpoint records
    sqlx::query("DELETE FROM checkpoints WHERE experiment_id = ?")
        .bind(exp_id)
        .execute(pool)
        .await?;

    // Delete checkpoint files from disk (best-effort)
    for path in checkpoint_paths {
        if let Err(e) = std::fs::remove_file(&path) {
            tracing::warn!("Failed to delete checkpoint file {}: {}", path, e);
        } else {
            tracing::info!("Deleted checkpoint file: {}", path);
        }
    }

    Ok(())
}

/// Update flow PID (called by worker when starting a flow)
pub async fn update_flow_pid(pool: &DbPool, id: i64, pid: Option<i64>) -> Result<bool> {
    let result = sqlx::query("UPDATE flows SET pid = ? WHERE id = ?")
        .bind(pid)
        .bind(id)
        .execute(pool)
        .await?;
    Ok(result.rows_affected() > 0)
}

/// Update flow heartbeat (called periodically by worker)
pub async fn update_flow_heartbeat(pool: &DbPool, id: i64) -> Result<bool> {
    let now = Utc::now().to_rfc3339();
    // A heartbeat is proof of a live runner: if some racing path (stale
    // reaper, crash-recovery) silently requeued the flow while its runner
    // kept working, self-heal back to 'running'. ONLY queued→running:
    // 'paused' must stay paused (pause is polled between generations, so a
    // paused flow legitimately heartbeats until the runner notices), and
    // terminal states are never resurrected.
    let result = sqlx::query(
        "UPDATE flows SET last_heartbeat = ?, \
         status = CASE WHEN status = 'queued' THEN 'running' ELSE status END \
         WHERE id = ?",
    )
        .bind(&now)
        .bind(id)
        .execute(pool)
        .await?;
    Ok(result.rows_affected() > 0)
}

/// Find stale running flows (no heartbeat in the last N seconds)
/// Returns flows that should be re-queued
#[allow(dead_code)]
pub async fn find_stale_running_flows(pool: &DbPool, stale_seconds: i64) -> Result<Vec<Flow>> {
    let cutoff = (Utc::now() - chrono::Duration::seconds(stale_seconds)).to_rfc3339();
    // NULL last_heartbeat is NOT instantly stale: a freshly-started runner's
    // heartbeat thread waits one full HEARTBEAT_INTERVAL (30s) before its
    // first beat, so treating NULL as stale silently requeued LIVE flows on
    // the reaper's next tick — the recurring "flow stuck 'queued' while the
    // runner works" bug (3 hits 03-05/07/2026). Fall back to started_at
    // (then created_at) under the SAME cutoff — parity with the worker-side
    // _recover_stale_flows logic.
    let rows = sqlx::query(
        r#"SELECT id, name, description, config_json, created_at, started_at, completed_at, status, seed_checkpoint_id, pid, last_heartbeat, status_message, pause_requested
           FROM flows
           WHERE status = 'running'
           AND COALESCE(last_heartbeat, started_at, created_at) < ?
           ORDER BY created_at ASC"#,
    )
    .bind(&cutoff)
    .fetch_all(pool)
    .await?;

    let mut flows = Vec::with_capacity(rows.len());
    for row in rows {
        flows.push(row_to_flow(&row)?);
    }
    Ok(flows)
}

/// Re-queue a stale flow (reset status and clear pid/heartbeat).
/// Guarded `WHERE status = 'running'`: data is preserved; the next worker
/// pickup resumes from the per-gen checkpoint. Wired into a background
/// task in main.rs since P3 (was dead code while the worker stale-FAILED
/// flows instead).
pub async fn requeue_stale_flow(pool: &DbPool, id: i64) -> Result<bool> {
    let result = sqlx::query(
        "UPDATE flows SET status = 'queued', pid = NULL, last_heartbeat = NULL WHERE id = ? AND status = 'running'"
    )
    .bind(id)
    .execute(pool)
    .await?;
    Ok(result.rows_affected() > 0)
}

/// Update flow for restart: set status to queued, clear pid, optionally clear seed
/// If from_beginning is true, deletes all linked experiments and their data,
/// then recreates fresh pending experiments from the saved metadata
pub async fn update_flow_for_restart(
    pool: &DbPool,
    id: i64,
    clear_seed: Option<Option<i64>>,
) -> Result<bool> {
    // If clearing seed (restart from beginning), snapshot experiments, delete, then recreate
    if clear_seed.is_some() {
        // Snapshot experiment metadata before deletion
        let saved_experiments: Vec<(String, i32, Option<String>, i32)> = sqlx::query_as(
            "SELECT name, sequence_order, phase_type, max_iterations FROM experiments WHERE flow_id = ? ORDER BY sequence_order"
        )
        .bind(id)
        .fetch_all(pool)
        .await?;

        // Delete all experiment data and experiments
        delete_flow_data(pool, id).await?;

        // Recreate fresh pending experiments from snapshot
        if !saved_experiments.is_empty() {
            // Get the flow config for create_pending_experiment
            let config_json: String = sqlx::query_scalar(
                "SELECT config_json FROM flows WHERE id = ?"
            )
            .bind(id)
            .fetch_one(pool)
            .await?;
            let flow_config: crate::models::FlowConfig = serde_json::from_str(&config_json)?;

            for (name, sequence_order, phase_type, _old_max_iterations) in &saved_experiments {
                // Recompute max_iterations from current flow config instead of using stale DB values
                let max_iterations = compute_max_iterations_from_phase_type(phase_type.as_deref(), &flow_config);
                create_pending_experiment(
                    pool,
                    name,
                    id,
                    *sequence_order,
                    phase_type.as_deref(),
                    max_iterations,
                    &flow_config,
                    None,  // params not preserved on restart
                ).await?;
            }
        }
    }

    if let Some(seed_id) = clear_seed {
        // Clear both pid and seed checkpoint
        let result = sqlx::query(
            "UPDATE flows SET status = 'queued', pid = NULL, seed_checkpoint_id = ?, started_at = NULL, completed_at = NULL WHERE id = ?"
        )
        .bind(seed_id)
        .bind(id)
        .execute(pool)
        .await?;
        Ok(result.rows_affected() > 0)
    } else {
        // Resume: clear display data (iterations, genomes, etc.) for non-completed experiments
        // so the dashboard shows a clean slate. Checkpoints are preserved for seeding.
        let non_completed_exp_ids: Vec<i64> = sqlx::query_scalar(
            "SELECT id FROM experiments WHERE flow_id = ? AND status != 'completed'"
        )
        .bind(id)
        .fetch_all(pool)
        .await?;

        for exp_id in &non_completed_exp_ids {
            clear_experiment_display_data(pool, *exp_id).await?;
        }

        // Reset non-completed experiments' timestamps
        sqlx::query(
            "UPDATE experiments SET started_at = NULL, ended_at = NULL, status = 'pending' WHERE flow_id = ? AND status != 'completed'"
        )
        .bind(id)
        .execute(pool)
        .await?;

        // Recompute max_iterations from current flow config for ALL experiments
        // (flow config may have been edited since experiments were created)
        let config_json: String = sqlx::query_scalar(
            "SELECT config_json FROM flows WHERE id = ?"
        )
        .bind(id)
        .fetch_one(pool)
        .await?;
        let flow_config: crate::models::FlowConfig = serde_json::from_str(&config_json)?;

        let all_experiments: Vec<(i64, Option<String>)> = sqlx::query_as(
            "SELECT id, phase_type FROM experiments WHERE flow_id = ?"
        )
        .bind(id)
        .fetch_all(pool)
        .await?;

        for (exp_id, phase_type) in &all_experiments {
            if let Some(max_iters) = compute_max_iterations_from_phase_type(phase_type.as_deref(), &flow_config) {
                sqlx::query("UPDATE experiments SET max_iterations = ? WHERE id = ?")
                    .bind(max_iters)
                    .bind(exp_id)
                    .execute(pool)
                    .await?;
            }
        }

        // Just reset status and pid
        let result = sqlx::query(
            "UPDATE flows SET status = 'queued', pid = NULL, started_at = NULL, completed_at = NULL WHERE id = ?"
        )
        .bind(id)
        .execute(pool)
        .await?;
        Ok(result.rows_affected() > 0)
    }
}
