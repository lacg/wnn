//! Iteration and genome-evaluation queries (split from db/mod.rs `queries`).

use super::*;
use super::queries::*;

/// Get iterations for an experiment
#[allow(dead_code)]
pub async fn get_experiment_iterations(pool: &DbPool, experiment_id: i64) -> Result<Vec<Iteration>> {
    let rows = sqlx::query(
        r#"SELECT id, experiment_id, iteration_num, best_ce, best_accuracy, avg_ce, avg_accuracy,
                  elite_count, offspring_count, offspring_viable, fitness_threshold,
                  elapsed_secs, baseline_ce, delta_baseline, delta_previous,
                  patience_counter, patience_max, candidates_total, created_at
           FROM iterations WHERE experiment_id = ? ORDER BY iteration_num"#,
    )
    .bind(experiment_id)
    .fetch_all(pool)
    .await?;

    let mut iterations = Vec::with_capacity(rows.len());
    for row in rows {
        iterations.push(row_to_iteration(&row)?);
    }
    Ok(iterations)
}

/// Get recent iterations for an experiment
pub async fn get_recent_iterations(pool: &DbPool, experiment_id: i64, limit: i32) -> Result<Vec<Iteration>> {
    let rows = sqlx::query(
        r#"SELECT id, experiment_id, iteration_num, best_ce, best_accuracy, avg_ce,
                  avg_accuracy, best_f1, best_fpr, mean_attitude_error_deg,
                  elite_count, offspring_count, offspring_viable,
                  fitness_threshold, elapsed_secs, baseline_ce, delta_baseline,
                  delta_previous, patience_counter, patience_max, candidates_total,
                  created_at
           FROM iterations
           WHERE experiment_id = ?
           ORDER BY created_at DESC
           LIMIT ?"#,
    )
    .bind(experiment_id)
    .bind(limit)
    .fetch_all(pool)
    .await?;

    let mut iterations = Vec::with_capacity(rows.len());
    for row in rows {
        iterations.push(row_to_iteration(&row)?);
    }
    // Reverse to get chronological order
    iterations.reverse();
    Ok(iterations)
}

fn row_to_iteration(row: &sqlx::sqlite::SqliteRow) -> Result<Iteration> {
    Ok(Iteration {
        id: row.get("id"),
        experiment_id: row.get("experiment_id"),
        iteration_num: row.get("iteration_num"),
        best_ce: row.get("best_ce"),
        best_accuracy: row.get("best_accuracy"),
        avg_ce: row.get("avg_ce"),
        avg_accuracy: row.get("avg_accuracy"),
        best_f1: row.try_get("best_f1").ok().flatten(),
        best_fpr: row.try_get("best_fpr").ok().flatten(),
        mean_attitude_error_deg: row.try_get("mean_attitude_error_deg").ok().flatten(),
        elite_count: row.get("elite_count"),
        offspring_count: row.get("offspring_count"),
        offspring_viable: row.get("offspring_viable"),
        fitness_threshold: row.get("fitness_threshold"),
        elapsed_secs: row.get("elapsed_secs"),
        baseline_ce: row.get("baseline_ce"),
        delta_baseline: row.get("delta_baseline"),
        delta_previous: row.get("delta_previous"),
        // Old rows store INTEGER, magnitude-aware patience (11/07/2026) stores
        // REAL fractions — decode either (the i32-only read made the whole
        // endpoint stream-reset on any post-magnitude flow).
        patience_counter: row
            .try_get::<Option<f64>, _>("patience_counter")
            .or_else(|_| row.try_get::<Option<i64>, _>("patience_counter").map(|v| v.map(|x| x as f64)))?,
        patience_max: row.get("patience_max"),
        candidates_total: row.get("candidates_total"),
        created_at: parse_datetime(row.get("created_at"))?,
    })
}

/// Get genome evaluations for an iteration
pub async fn get_genome_evaluations(pool: &DbPool, iteration_id: i64) -> Result<Vec<GenomeEvaluation>> {
    let rows = sqlx::query(
        r#"SELECT ge.id, ge.iteration_id, ge.genome_id, ge.position, ge.role,
                  ge.elite_rank, ge.ce, ge.accuracy, ge.fitness_score,
                  ge.f1_macro, ge.fpr, ge.eval_time_ms,
                  ge.created_at, g.tiers_json
           FROM genome_evaluations ge
           LEFT JOIN genomes g ON ge.genome_id = g.id
           WHERE ge.iteration_id = ?
           ORDER BY ge.position"#,
    )
    .bind(iteration_id)
    .fetch_all(pool)
    .await?;

    let mut evaluations = Vec::with_capacity(rows.len());
    for row in rows {
        evaluations.push(GenomeEvaluation {
            id: row.get("id"),
            iteration_id: row.get("iteration_id"),
            genome_id: row.get("genome_id"),
            position: row.get("position"),
            role: parse_genome_role(row.get::<String, _>("role").as_str()),
            elite_rank: row.get("elite_rank"),
            ce: row.get("ce"),
            accuracy: row.get("accuracy"),
            fitness_score: row.get("fitness_score"),
            f1_macro: row.get("f1_macro"),
            fpr: row.get("fpr"),
            eval_time_ms: row.get("eval_time_ms"),
            created_at: parse_datetime(row.get("created_at"))?,
            tiers_json: row.get("tiers_json"),
        });
    }
    Ok(evaluations)
}

fn parse_genome_role(s: &str) -> GenomeRole {
    match s {
        "elite" => GenomeRole::Elite,
        "offspring" => GenomeRole::Offspring,
        "init" => GenomeRole::Init,
        "top_k" => GenomeRole::TopK,
        "neighbor" => GenomeRole::Neighbor,
        "current" => GenomeRole::Current,
        _ => GenomeRole::Offspring,
    }
}
