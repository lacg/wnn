//! Checkpoint queries (split from db/mod.rs `queries`).

use super::queries::*;
use super::*;

// =============================================================================
// Checkpoint queries
// =============================================================================

pub async fn list_checkpoints(
	pool: &DbPool,
	experiment_id: Option<i64>,
	checkpoint_type: Option<&str>,
	limit: i32,
	offset: i32,
) -> Result<Vec<Checkpoint>>
{
	let mut query = String::from(
		r#"SELECT c.id, c.experiment_id, c.iteration_id, c.name, c.file_path, c.file_size_bytes,
                  c.checkpoint_type, c.best_ce, c.best_accuracy, c.genome_stats_json, c.created_at,
                  e.flow_id, f.name as flow_name
           FROM checkpoints c
           LEFT JOIN experiments e ON c.experiment_id = e.id
           LEFT JOIN flows f ON e.flow_id = f.id
           WHERE 1=1"#,
	);

	if experiment_id.is_some()
	{
		query.push_str(" AND c.experiment_id = ?");
	}
	if checkpoint_type.is_some()
	{
		query.push_str(" AND c.checkpoint_type = ?");
	}
	query.push_str(" ORDER BY c.created_at DESC LIMIT ? OFFSET ?");

	let mut q = sqlx::query(&query);

	if let Some(exp_id) = experiment_id
	{
		q = q.bind(exp_id);
	}
	if let Some(cp_type) = checkpoint_type
	{
		q = q.bind(cp_type);
	}
	q = q.bind(limit).bind(offset);

	let rows = q.fetch_all(pool).await?;

	let mut checkpoints = Vec::with_capacity(rows.len());
	for row in rows
	{
		checkpoints.push(row_to_checkpoint_with_flow(&row)?);
	}
	Ok(checkpoints)
}

pub async fn get_checkpoint(pool: &DbPool, id: i64) -> Result<Option<Checkpoint>>
{
	let row = sqlx::query(
		r#"SELECT id, experiment_id, iteration_id, name, file_path, file_size_bytes,
                  checkpoint_type, best_ce, best_accuracy, genome_stats_json, created_at
           FROM checkpoints WHERE id = ?"#,
	)
	.bind(id)
	.fetch_optional(pool)
	.await?;

	match row
	{
		Some(r) => Ok(Some(row_to_checkpoint(&r)?)),
		None => Ok(None),
	}
}

pub async fn create_checkpoint(
	pool: &DbPool,
	experiment_id: i64,
	name: &str,
	file_path: &str,
	checkpoint_type: &str,
	file_size_bytes: Option<i64>,
	iteration_id: Option<i64>,
	best_ce: Option<f64>,
	best_accuracy: Option<f64>,
	genome_stats: Option<&serde_json::Value>,
) -> Result<i64>
{
	let now = Utc::now().to_rfc3339();
	let genome_stats_json = genome_stats.map(|v| serde_json::to_string(v).unwrap_or_default());

	let result = sqlx::query(
		r#"INSERT INTO checkpoints
           (experiment_id, iteration_id, name, file_path, file_size_bytes,
            checkpoint_type, best_ce, best_accuracy, genome_stats_json, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"#,
	)
	.bind(experiment_id)
	.bind(iteration_id)
	.bind(name)
	.bind(file_path)
	.bind(file_size_bytes)
	.bind(checkpoint_type)
	.bind(best_ce)
	.bind(best_accuracy)
	.bind(&genome_stats_json)
	.bind(&now)
	.execute(pool)
	.await?;

	Ok(result.last_insert_rowid())
}

pub async fn delete_checkpoint(pool: &DbPool, id: i64) -> Result<(bool, Option<String>)>
{
	let row = sqlx::query("SELECT file_path FROM checkpoints WHERE id = ?")
		.bind(id)
		.fetch_optional(pool)
		.await?;

	let Some(row) = row
	else
	{
		return Ok((false, None));
	};

	let file_path: String = row.get("file_path");

	// Delete checkpoint
	let result = sqlx::query("DELETE FROM checkpoints WHERE id = ?")
		.bind(id)
		.execute(pool)
		.await?;

	Ok((result.rows_affected() > 0, Some(file_path)))
}

fn row_to_checkpoint(row: &sqlx::sqlite::SqliteRow) -> Result<Checkpoint>
{
	let checkpoint_type_str: String = row.get("checkpoint_type");
	let genome_stats_json: Option<String> = row.get("genome_stats_json");
	let genome_stats = genome_stats_json.and_then(|s| serde_json::from_str(&s).ok());

	Ok(Checkpoint {
		id: row.get("id"),
		experiment_id: row.get("experiment_id"),
		iteration_id: row.get("iteration_id"),
		name: row.get("name"),
		file_path: row.get("file_path"),
		file_size_bytes: row.get("file_size_bytes"),
		checkpoint_type: parse_checkpoint_type(&checkpoint_type_str),
		best_ce: row.get("best_ce"),
		best_accuracy: row.get("best_accuracy"),
		genome_stats,
		created_at: parse_datetime(row.get("created_at"))?,
		flow_id: None,
		flow_name: None,
	})
}

fn row_to_checkpoint_with_flow(row: &sqlx::sqlite::SqliteRow) -> Result<Checkpoint>
{
	let checkpoint_type_str: String = row.get("checkpoint_type");
	let genome_stats_json: Option<String> = row.get("genome_stats_json");
	let genome_stats = genome_stats_json.and_then(|s| serde_json::from_str(&s).ok());

	Ok(Checkpoint {
		id: row.get("id"),
		experiment_id: row.get("experiment_id"),
		iteration_id: row.get("iteration_id"),
		name: row.get("name"),
		file_path: row.get("file_path"),
		file_size_bytes: row.get("file_size_bytes"),
		checkpoint_type: parse_checkpoint_type(&checkpoint_type_str),
		best_ce: row.get("best_ce"),
		best_accuracy: row.get("best_accuracy"),
		genome_stats,
		created_at: parse_datetime(row.get("created_at"))?,
		flow_id: row.get("flow_id"),
		flow_name: row.get("flow_name"),
	})
}

fn parse_checkpoint_type(s: &str) -> CheckpointType
{
	match s
	{
		"auto" => CheckpointType::Auto,
		"user" => CheckpointType::User,
		"experiment_end" => CheckpointType::ExperimentEnd,
		_ => CheckpointType::Auto,
	}
}
