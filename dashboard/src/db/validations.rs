//! Validation summary, cached validation, and combined validation queries (split from db/mod.rs `queries`).

use super::queries::*;
use super::*;

// =============================================================================
// Validation Summary queries
// =============================================================================

/// Get validation summaries for an experiment
pub async fn get_validation_summaries(
	pool: &DbPool,
	experiment_id: i64,
) -> Result<Vec<ValidationSummary>>
{
	let rows = sqlx::query(
		r#"SELECT id, flow_id, experiment_id, validation_point, genome_type,
                  genome_hash, ce, accuracy, f1_macro, fpr, threshold_metadata, created_at
           FROM validation_summaries
           WHERE experiment_id = ?
           ORDER BY validation_point, genome_type"#,
	)
	.bind(experiment_id)
	.fetch_all(pool)
	.await?;

	let mut summaries = Vec::with_capacity(rows.len());
	for row in rows
	{
		summaries.push(row_to_validation_summary(&row)?);
	}
	Ok(summaries)
}

/// Get validation summaries for a flow (all experiments)
pub async fn get_flow_validation_summaries(
	pool: &DbPool,
	flow_id: i64,
) -> Result<Vec<ValidationSummary>>
{
	let rows = sqlx::query(
        r#"SELECT vs.id, vs.flow_id, vs.experiment_id, vs.validation_point, vs.genome_type,
                  vs.genome_hash, vs.ce, vs.accuracy, vs.f1_macro, vs.fpr, vs.threshold_metadata, vs.created_at
           FROM validation_summaries vs
           JOIN experiments e ON vs.experiment_id = e.id
           WHERE e.flow_id = ?
           ORDER BY e.sequence_order, vs.validation_point, vs.genome_type"#,
    )
    .bind(flow_id)
    .fetch_all(pool)
    .await?;

	let mut summaries = Vec::with_capacity(rows.len());
	for row in rows
	{
		summaries.push(row_to_validation_summary(&row)?);
	}
	Ok(summaries)
}

/// Check if a genome has already been validated (by genome_hash)
/// Returns the cached CE, accuracy, f1_macro, fpr, and threshold_metadata if found.
/// When dataset_key is provided, only matches validations from flows with the same
/// dataset+encoding config to prevent cross-dataset cache poisoning.
pub async fn get_cached_validation(
	pool: &DbPool,
	genome_hash: &str,
	dataset_key: Option<&str>,
) -> Result<
	Option<(
		f64,
		f64,
		Option<f64>,
		Option<f64>,
		Option<serde_json::Value>,
	)>,
>
{
	let row = if let Some(dk) = dataset_key
	{
		// Mirror the Python construction in worker.py (build_dataset_key):
		//   "{ds}_{nb}b_{sp}{_raw?}{_inv-<mode>?}{_oi0|_oi1}"
		// Each suffix is appended only when the corresponding flag is set, so cache
		// entries are scoped to dataset + bits + split + raw-mode + invalid-encoding +
		// training-algo. Without this, paired flows that differ only in training algo
		// (e.g. WNN_ORDER_INDEPENDENT_TRAIN) collide in cache.
		sqlx::query(
            r#"SELECT vs.ce, vs.accuracy, vs.f1_macro, vs.fpr, vs.threshold_metadata
               FROM validation_summaries vs
               JOIN flows f ON vs.flow_id = f.id
               WHERE vs.genome_hash = ?
                 AND (json_extract(f.config_json, '$.params.ids_dataset') || '_' ||
                      json_extract(f.config_json, '$.params.ids_n_bits') || 'b_' ||
                      json_extract(f.config_json, '$.params.ids_split') ||
                      CASE WHEN json_extract(f.config_json, '$.params.ids_raw') = 1
                           THEN '_raw' ELSE '' END ||
                      CASE WHEN json_extract(f.config_json, '$.params.ids_invalid_encoding') IS NOT NULL
                            AND json_extract(f.config_json, '$.params.ids_invalid_encoding') != 'none'
                           THEN '_inv-' || json_extract(f.config_json, '$.params.ids_invalid_encoding')
                           ELSE '' END ||
                      CASE WHEN json_extract(f.config_json, '$.params.wnn_order_independent_train') = 1
                           THEN '_oi1' ELSE '_oi0' END) = ?
               ORDER BY vs.threshold_metadata IS NOT NULL DESC
               LIMIT 1"#,
        )
        .bind(genome_hash)
        .bind(dk)
        .fetch_optional(pool)
        .await?
	}
	else
	{
		sqlx::query(
			r#"SELECT ce, accuracy, f1_macro, fpr, threshold_metadata
               FROM validation_summaries WHERE genome_hash = ?
               ORDER BY threshold_metadata IS NOT NULL DESC
               LIMIT 1"#,
		)
		.bind(genome_hash)
		.fetch_optional(pool)
		.await?
	};

	match row
	{
		Some(r) =>
		{
			let tm_str: Option<String> = r.get("threshold_metadata");
			let tm = tm_str.and_then(|s| serde_json::from_str(&s).ok());
			Ok(Some((
				r.get("ce"),
				r.get("accuracy"),
				r.get("f1_macro"),
				r.get("fpr"),
				tm,
			)))
		}
		None => Ok(None),
	}
}

/// Create a validation summary (upsert by experiment_id + validation_point + genome_type)
pub async fn upsert_validation_summary(
	pool: &DbPool,
	flow_id: Option<i64>,
	experiment_id: i64,
	validation_point: &str,
	genome_type: &str,
	genome_hash: &str,
	ce: f64,
	accuracy: f64,
	f1_macro: Option<f64>,
	fpr: Option<f64>,
	threshold_metadata: Option<&str>,
) -> Result<i64>
{
	let now = Utc::now().to_rfc3339();

	let result = sqlx::query(
        r#"INSERT INTO validation_summaries
           (flow_id, experiment_id, validation_point, genome_type, genome_hash, ce, accuracy, f1_macro, fpr, threshold_metadata, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(experiment_id, validation_point, genome_type) DO UPDATE SET
             flow_id = excluded.flow_id,
             genome_hash = excluded.genome_hash,
             ce = excluded.ce,
             accuracy = excluded.accuracy,
             f1_macro = excluded.f1_macro,
             fpr = excluded.fpr,
             threshold_metadata = excluded.threshold_metadata,
             created_at = excluded.created_at"#,
    )
    .bind(flow_id)
    .bind(experiment_id)
    .bind(validation_point)
    .bind(genome_type)
    .bind(genome_hash)
    .bind(ce)
    .bind(accuracy)
    .bind(f1_macro)
    .bind(fpr)
    .bind(threshold_metadata)
    .bind(&now)
    .execute(pool)
    .await?;

	Ok(result.last_insert_rowid())
}

fn row_to_validation_summary(row: &sqlx::sqlite::SqliteRow) -> Result<ValidationSummary>
{
	let validation_point_str: String = row.get("validation_point");
	let genome_type_str: String = row.get("genome_type");
	let threshold_metadata_str: Option<String> = row.get("threshold_metadata");
	let threshold_metadata = threshold_metadata_str.and_then(|s| serde_json::from_str(&s).ok());

	Ok(ValidationSummary {
		id: row.get("id"),
		flow_id: row.get("flow_id"),
		experiment_id: row.get("experiment_id"),
		validation_point: parse_validation_point(&validation_point_str),
		genome_type: parse_genome_validation_type(&genome_type_str),
		genome_hash: row.get("genome_hash"),
		ce: row.get("ce"),
		accuracy: row.get("accuracy"),
		f1_macro: row.get("f1_macro"),
		fpr: row.get("fpr"),
		threshold_metadata,
		created_at: parse_datetime(row.get("created_at"))?,
	})
}

fn parse_validation_point(s: &str) -> ValidationPoint
{
	match s
	{
		"init" => ValidationPoint::Init,
		"final" => ValidationPoint::Final,
		_ => ValidationPoint::Final,
	}
}

fn parse_genome_validation_type(s: &str) -> GenomeValidationType
{
	match s
	{
		"best_ce" => GenomeValidationType::BestCe,
		"best_acc" => GenomeValidationType::BestAcc,
		"best_f1" => GenomeValidationType::BestF1,
		"best_fpr" => GenomeValidationType::BestFpr,
		"best_fitness" => GenomeValidationType::BestFitness,
		"best_overall_ce" => GenomeValidationType::BestOverallCe,
		"best_overall_acc" => GenomeValidationType::BestOverallAcc,
		_ => GenomeValidationType::BestCe,
	}
}

// =============================================================================
// Combined Validation queries (multi-stage end-to-end metrics)
// =============================================================================

pub async fn get_combined_validations(
	pool: &DbPool,
	flow_id: i64,
) -> Result<Vec<CombinedValidation>>
{
	let rows = sqlx::query(
		r#"SELECT id, flow_id, genome_type, combined_ce, combined_accuracy,
                  per_stage_ce_json, per_stage_acc_json, unigram_lambda, created_at
           FROM combined_validations
           WHERE flow_id = ?
           ORDER BY genome_type"#,
	)
	.bind(flow_id)
	.fetch_all(pool)
	.await?;

	let mut results = Vec::with_capacity(rows.len());
	for row in rows
	{
		results.push(row_to_combined_validation(&row)?);
	}
	Ok(results)
}

pub async fn upsert_combined_validation(
	pool: &DbPool,
	flow_id: i64,
	genome_type: &str,
	combined_ce: f64,
	combined_accuracy: f64,
	per_stage_ce: Option<&[f64]>,
	per_stage_acc: Option<&[f64]>,
	unigram_lambda: Option<f64>,
) -> Result<i64>
{
	let now = Utc::now().to_rfc3339();
	let per_stage_ce_json = per_stage_ce.map(|v| serde_json::to_string(v).unwrap_or_default());
	let per_stage_acc_json = per_stage_acc.map(|v| serde_json::to_string(v).unwrap_or_default());

	let result = sqlx::query(
        r#"INSERT INTO combined_validations
           (flow_id, genome_type, combined_ce, combined_accuracy, per_stage_ce_json, per_stage_acc_json, unigram_lambda, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(flow_id, genome_type) DO UPDATE SET
             combined_ce = excluded.combined_ce,
             combined_accuracy = excluded.combined_accuracy,
             per_stage_ce_json = excluded.per_stage_ce_json,
             per_stage_acc_json = excluded.per_stage_acc_json,
             unigram_lambda = excluded.unigram_lambda,
             created_at = excluded.created_at"#,
    )
    .bind(flow_id)
    .bind(genome_type)
    .bind(combined_ce)
    .bind(combined_accuracy)
    .bind(&per_stage_ce_json)
    .bind(&per_stage_acc_json)
    .bind(unigram_lambda)
    .bind(&now)
    .execute(pool)
    .await?;

	Ok(result.last_insert_rowid())
}

fn row_to_combined_validation(row: &sqlx::sqlite::SqliteRow) -> Result<CombinedValidation>
{
	let genome_type_str: String = row.get("genome_type");
	let per_stage_ce_json: Option<String> = row.get("per_stage_ce_json");
	let per_stage_ce =
		per_stage_ce_json.and_then(|json| serde_json::from_str::<Vec<f64>>(&json).ok());
	let per_stage_acc_json: Option<String> = row.get("per_stage_acc_json");
	let per_stage_acc =
		per_stage_acc_json.and_then(|json| serde_json::from_str::<Vec<f64>>(&json).ok());

	let unigram_lambda: Option<f64> = row.try_get("unigram_lambda").unwrap_or(None);

	Ok(CombinedValidation {
		id: row.get("id"),
		flow_id: row.get("flow_id"),
		genome_type: parse_genome_validation_type(&genome_type_str),
		combined_ce: row.get("combined_ce"),
		combined_accuracy: row.get("combined_accuracy"),
		per_stage_ce,
		per_stage_acc,
		unigram_lambda,
		created_at: parse_datetime(row.get("created_at"))?,
	})
}
