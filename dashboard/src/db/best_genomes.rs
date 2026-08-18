//! Best-genome (leaderboard) queries (split from db/mod.rs `queries`).

use super::queries::*;
use super::*;

// =============================================================================
// Best Genomes (Leaderboard) queries
// =============================================================================

/// List best genomes with optional filters, joined with genomes table for architecture info
pub async fn list_best_genomes(
	pool: &DbPool,
	task_type: Option<&str>,
	stage: Option<&str>,
	metric: Option<&str>,
	limit: i32,
	offset: i32,
	feature_selection: Option<&str>,
	n_bits: Option<i32>,
	ids_dataset: Option<&str>,
	ids_split: Option<&str>,
) -> Result<Vec<BestGenome>>
{
	let rows = sqlx::query(
		r#"SELECT bg.id, bg.task_type, bg.stage, bg.metric,
                  bg.genome_id, bg.genome_hash, bg.rank,
                  bg.ce, bg.accuracy, bg.f1_macro, bg.fpr,
                  bg.flow_id, bg.experiment_id,
                  bg.threshold_mode,
                  bg.hf_repo_id, bg.hf_exported_at,
                  bg.created_at, bg.updated_at,
                  g.tiers_json, g.total_clusters, g.total_neurons, g.architecture_type
           FROM best_genomes bg
           LEFT JOIN genomes g ON g.id = bg.genome_id
           LEFT JOIN flows f ON f.id = bg.flow_id
           WHERE (? IS NULL OR bg.task_type = ?)
             AND (? IS NULL OR bg.stage = ?)
             AND (? IS NULL OR bg.metric = ?)
             AND (? IS NULL OR json_extract(f.config_json, '$.params.ids_feature_selection') = ?)
             AND (? IS NULL OR json_extract(f.config_json, '$.params.ids_n_bits') = ?)
             AND (? IS NULL OR json_extract(f.config_json, '$.params.ids_dataset') = ?)
             AND (? IS NULL OR json_extract(f.config_json, '$.params.ids_split') = ?)
           ORDER BY bg.rank ASC NULLS LAST, bg.f1_macro DESC NULLS LAST, bg.ce ASC
           LIMIT ? OFFSET ?"#,
	)
	.bind(task_type)
	.bind(task_type)
	.bind(stage)
	.bind(stage)
	.bind(metric)
	.bind(metric)
	.bind(feature_selection)
	.bind(feature_selection)
	.bind(n_bits)
	.bind(n_bits)
	.bind(ids_dataset)
	.bind(ids_dataset)
	.bind(ids_split)
	.bind(ids_split)
	.bind(limit)
	.bind(offset)
	.fetch_all(pool)
	.await?;

	let mut results = Vec::with_capacity(rows.len());
	for row in rows
	{
		results.push(row_to_best_genome(&row)?);
	}
	Ok(results)
}

/// Get a single best genome entry by ID
pub async fn get_best_genome(pool: &DbPool, id: i64) -> Result<Option<BestGenome>>
{
	let row = sqlx::query(
		r#"SELECT bg.id, bg.task_type, bg.stage, bg.metric,
                  bg.genome_id, bg.genome_hash, bg.rank,
                  bg.ce, bg.accuracy, bg.f1_macro, bg.fpr,
                  bg.flow_id, bg.experiment_id,
                  bg.threshold_mode,
                  bg.hf_repo_id, bg.hf_exported_at,
                  bg.created_at, bg.updated_at,
                  g.tiers_json, g.total_clusters, g.total_neurons, g.architecture_type
           FROM best_genomes bg
           LEFT JOIN genomes g ON g.id = bg.genome_id
           WHERE bg.id = ?"#,
	)
	.bind(id)
	.fetch_optional(pool)
	.await?;

	match row
	{
		Some(r) => Ok(Some(row_to_best_genome(&r)?)),
		None => Ok(None),
	}
}

/// Get genome data (connections_json) for download
pub async fn get_best_genome_data(pool: &DbPool, id: i64) -> Result<Option<serde_json::Value>>
{
	let row = sqlx::query(
		r#"SELECT g.connections_json, g.tiers_json, g.total_clusters, g.total_neurons,
                  g.architecture_type, g.hf_config_json, bg.genome_hash,
                  bg.ce, bg.accuracy, bg.f1_macro, bg.fpr
           FROM best_genomes bg
           JOIN genomes g ON g.id = bg.genome_id
           WHERE bg.id = ?"#,
	)
	.bind(id)
	.fetch_optional(pool)
	.await?;

	match row
	{
		Some(r) =>
		{
			let connections_json: Option<String> = r.get("connections_json");
			let tiers_json: String = r.get("tiers_json");
			let total_clusters: i32 = r.get("total_clusters");
			let total_neurons: i32 = r.get("total_neurons");
			let architecture_type: Option<String> = r.get("architecture_type");
			let hf_config_json: Option<String> = r.get("hf_config_json");
			let genome_hash: String = r.get("genome_hash");
			let ce: f64 = r.get("ce");
			let accuracy: f64 = r.get("accuracy");
			let f1_macro: Option<f64> = r.get("f1_macro");
			let fpr: Option<f64> = r.get("fpr");

			Ok(Some(serde_json::json!({
					"genome_hash": genome_hash,
					"connections_json": connections_json,
					"tiers_json": tiers_json,
					"total_clusters": total_clusters,
					"total_neurons": total_neurons,
					"architecture_type": architecture_type,
					"hf_config_json": hf_config_json,
					"ce": ce,
					"accuracy": accuracy,
					"f1_macro": f1_macro,
					"fpr": fpr,
			})))
		}
		None => Ok(None),
	}
}

/// Submit genome(s) to best_genomes leaderboard.
/// For each genome: find or create genomes row, then INSERT OR IGNORE into best_genomes.
pub async fn submit_best_genome(
	pool: &DbPool,
	task_type: &str,
	stage: &str,
	metric: &str,
	genome_hash: &str,
	ce: f64,
	accuracy: f64,
	f1_macro: Option<f64>,
	fpr: Option<f64>,
	flow_id: Option<i64>,
	experiment_id: Option<i64>,
	genome_data: Option<&serde_json::Value>,
) -> Result<Option<i64>>
{
	let now = Utc::now().to_rfc3339();

	// Step 1: Find or create genome row
	let genome_id = if let Some(data) = genome_data
	{
		let config_hash = data
			.get("config_hash")
			.and_then(|v| v.as_str())
			.unwrap_or(genome_hash);
		let tiers_json = data
			.get("tiers_json")
			.and_then(|v| v.as_str())
			.unwrap_or("[]");
		let total_clusters = data
			.get("total_clusters")
			.and_then(|v| v.as_i64())
			.unwrap_or(0) as i32;
		let total_neurons = data
			.get("total_neurons")
			.and_then(|v| v.as_i64())
			.unwrap_or(0) as i32;
		let architecture_type = data
			.get("architecture_type")
			.and_then(|v| v.as_str())
			.unwrap_or("bitwise");
		let connections_json = data.get("connections_json").and_then(|v| v.as_str());
		let exp_id = experiment_id.unwrap_or(0);

		// Try to find existing genome row by (experiment_id, config_hash)
		let existing =
			sqlx::query("SELECT id FROM genomes WHERE experiment_id = ? AND config_hash = ?")
				.bind(exp_id)
				.bind(config_hash)
				.fetch_optional(pool)
				.await?;

		if let Some(row) = existing
		{
			let gid: i64 = row.get("id");
			// Update genome_hash and connections_json if not yet populated
			let _ = sqlx::query(
				r#"UPDATE genomes SET genome_hash = ?, connections_json = COALESCE(connections_json, ?)
                   WHERE id = ? AND (genome_hash IS NULL OR connections_json IS NULL)"#,
			)
			.bind(genome_hash)
			.bind(connections_json)
			.bind(gid)
			.execute(pool)
			.await;
			gid
		}
		else
		{
			// Insert new genome row
			let result = sqlx::query(
				r#"INSERT INTO genomes (experiment_id, config_hash, genome_hash, tiers_json,
                                       total_clusters, total_neurons, total_memory_bytes,
                                       architecture_type, connections_json, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, 0, ?, ?, ?)"#,
			)
			.bind(exp_id)
			.bind(config_hash)
			.bind(genome_hash)
			.bind(tiers_json)
			.bind(total_clusters)
			.bind(total_neurons)
			.bind(architecture_type)
			.bind(connections_json)
			.bind(&now)
			.execute(pool)
			.await?;
			result.last_insert_rowid()
		}
	}
	else
	{
		// No genome_data — look up by genome_hash in genomes table
		let row = sqlx::query("SELECT id FROM genomes WHERE genome_hash = ? LIMIT 1")
			.bind(genome_hash)
			.fetch_optional(pool)
			.await?;
		match row
		{
			Some(r) => r.get("id"),
			None => return Ok(None), // Can't create best_genome without a genome row
		}
	};

	// Step 2: INSERT OR IGNORE into best_genomes
	// Extract threshold_mode from genome_data if present
	let threshold_mode = genome_data
		.and_then(|d| d.get("threshold_mode"))
		.and_then(|v| v.as_str())
		.unwrap_or("train_cal");

	let result = sqlx::query(
		r#"INSERT OR IGNORE INTO best_genomes
           (task_type, stage, metric, genome_id, genome_hash,
            ce, accuracy, f1_macro, fpr, flow_id, experiment_id,
            threshold_mode, created_at, updated_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"#,
	)
	.bind(task_type)
	.bind(stage)
	.bind(metric)
	.bind(genome_id)
	.bind(genome_hash)
	.bind(ce)
	.bind(accuracy)
	.bind(f1_macro)
	.bind(fpr)
	.bind(flow_id)
	.bind(experiment_id)
	.bind(threshold_mode)
	.bind(&now)
	.bind(&now)
	.execute(pool)
	.await?;

	if result.rows_affected() == 0
	{
		// Already exists — update metrics if better
		let _ = sqlx::query(
            r#"UPDATE best_genomes SET
                 ce = MIN(ce, ?), accuracy = MAX(accuracy, ?),
                 f1_macro = COALESCE(MAX(f1_macro, ?), ?),
                 fpr = COALESCE(MIN(fpr, ?), ?),
                 updated_at = ?
               WHERE task_type = ? AND stage = ? AND metric = ? AND genome_hash = ? AND threshold_mode = ?"#,
        )
        .bind(ce).bind(accuracy)
        .bind(f1_macro).bind(f1_macro)
        .bind(fpr).bind(fpr)
        .bind(&now)
        .bind(task_type).bind(stage).bind(metric).bind(genome_hash).bind(threshold_mode)
        .execute(pool)
        .await;
		// Return existing id
		let row = sqlx::query(
            "SELECT id FROM best_genomes WHERE task_type = ? AND stage = ? AND metric = ? AND genome_hash = ?"
        )
        .bind(task_type).bind(stage).bind(metric).bind(genome_hash)
        .fetch_optional(pool)
        .await?;
		Ok(row.map(|r| r.get("id")))
	}
	else
	{
		Ok(Some(result.last_insert_rowid()))
	}
}

/// Recalculate rankings for a specific category.
/// Re-ranks all entries, prunes beyond max_entries.
pub async fn recalculate_rankings(
	pool: &DbPool,
	task_type: &str,
	stage: &str,
	metric: &str,
	max_entries: i32,
) -> Result<i32>
{
	// Determine sort order based on metric
	let order = match metric
	{
		"ce" => "ce ASC",
		"accuracy" | "f1_macro" => "accuracy DESC, ce ASC",
		_ => "ce ASC",
	};

	// Fetch all entries for this category in sorted order
	let query = format!(
		r#"SELECT id FROM best_genomes
           WHERE task_type = ? AND stage = ? AND metric = ?
           ORDER BY {}"#,
		order
	);
	let rows = sqlx::query(&query)
		.bind(task_type)
		.bind(stage)
		.bind(metric)
		.fetch_all(pool)
		.await?;

	let total = rows.len() as i32;
	let now = Utc::now().to_rfc3339();

	// Update ranks
	for (i, row) in rows.iter().enumerate()
	{
		let id: i64 = row.get("id");
		let rank = (i + 1) as i32;

		if rank <= max_entries
		{
			let _ = sqlx::query("UPDATE best_genomes SET rank = ?, updated_at = ? WHERE id = ?")
				.bind(rank)
				.bind(&now)
				.bind(id)
				.execute(pool)
				.await;
		}
		else
		{
			// Prune entries beyond max
			let _ = sqlx::query("DELETE FROM best_genomes WHERE id = ?")
				.bind(id)
				.execute(pool)
				.await;
		}
	}

	Ok(total.min(max_entries))
}

/// Delete a best genome entry
pub async fn delete_best_genome(pool: &DbPool, id: i64) -> Result<bool>
{
	let result = sqlx::query("DELETE FROM best_genomes WHERE id = ?")
		.bind(id)
		.execute(pool)
		.await?;
	Ok(result.rows_affected() > 0)
}

/// Update HuggingFace export info for a best genome
pub async fn update_best_genome_hf(pool: &DbPool, id: i64, hf_repo_id: &str) -> Result<bool>
{
	let now = Utc::now().to_rfc3339();
	let result = sqlx::query(
		"UPDATE best_genomes SET hf_repo_id = ?, hf_exported_at = ?, updated_at = ? WHERE id = ?",
	)
	.bind(hf_repo_id)
	.bind(&now)
	.bind(&now)
	.bind(id)
	.execute(pool)
	.await?;
	Ok(result.rows_affected() > 0)
}

fn row_to_best_genome(row: &sqlx::sqlite::SqliteRow) -> Result<BestGenome>
{
	Ok(BestGenome {
		id: row.get("id"),
		task_type: row.get("task_type"),
		stage: row.get("stage"),
		metric: row.get("metric"),
		genome_id: row.get("genome_id"),
		genome_hash: row.get("genome_hash"),
		rank: row.get("rank"),
		ce: row.get("ce"),
		accuracy: row.get("accuracy"),
		f1_macro: row.get("f1_macro"),
		fpr: row.get("fpr"),
		flow_id: row.get("flow_id"),
		experiment_id: row.get("experiment_id"),
		threshold_mode: row
			.try_get("threshold_mode")
			.unwrap_or("train_cal".to_string()),
		hf_repo_id: row.get("hf_repo_id"),
		hf_exported_at: row.get("hf_exported_at"),
		created_at: parse_datetime(row.get("created_at"))?,
		updated_at: parse_datetime(row.get("updated_at"))?,
		tiers_json: row.try_get("tiers_json").unwrap_or(None),
		total_clusters: row.try_get("total_clusters").unwrap_or(None),
		total_neurons: row.try_get("total_neurons").unwrap_or(None),
		architecture_type_str: row.try_get("architecture_type").unwrap_or(None),
	})
}
