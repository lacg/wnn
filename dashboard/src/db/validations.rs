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

/// Validation-cache scope stamped on every new validation_summaries row and
/// required for every cache hit (PAPER-CRITICAL fix, 10/10/2026).
///
/// `key` is built in ONE place — the worker's `validation_cache_key.py` — and
/// stored verbatim, so the dashboard never re-derives it from config_json (the
/// old SQL mirror drifted: it omitted memory_mode, feature selection,
/// classification and the trainer ABI, so a QUAD flow inherited a QSR flow's
/// TEST row). `worker_abi` pins the trainer version.
#[derive(Debug, Clone, Copy)]
pub struct CacheScope<'a>
{
	pub key: &'a str,
	pub worker_abi: i64,
}

impl<'a> CacheScope<'a>
{
	/// Both halves or nothing: a request missing either can never hit.
	pub fn from_parts(key: Option<&'a str>, worker_abi: Option<i64>) -> Option<Self>
	{
		match (key, worker_abi)
		{
			(Some(key), Some(worker_abi)) if !key.is_empty() => Some(Self { key, worker_abi }),
			_ => None,
		}
	}
}

type CachedValidation = (f64, f64, Option<f64>, Option<f64>, Option<serde_json::Value>);

/// Check if a genome has already been validated UNDER THE SAME SCOPE.
/// Returns the cached CE, accuracy, f1_macro, fpr, and threshold_metadata if found.
///
/// A hit requires genome_hash, cache_key AND worker_abi to match a row stamped
/// with them. No scope -> no hit; rows with a NULL key or ABI (every row written
/// before the fix, and every row from a pre-fix worker) are never served.
pub async fn get_cached_validation(
	pool: &DbPool,
	genome_hash: &str,
	scope: Option<CacheScope<'_>>,
) -> Result<Option<CachedValidation>>
{
	let Some(scope) = scope
	else
	{
		return Ok(None);
	};
	let row = sqlx::query(
		r#"SELECT ce, accuracy, f1_macro, fpr, threshold_metadata
           FROM validation_summaries
           WHERE genome_hash = ? AND cache_key = ? AND worker_abi = ?
           ORDER BY threshold_metadata IS NOT NULL DESC, id DESC
           LIMIT 1"#,
	)
	.bind(genome_hash)
	.bind(scope.key)
	.bind(scope.worker_abi)
	.fetch_optional(pool)
	.await?;
	Ok(row.map(|r| cached_validation_from_row(&r)))
}

fn cached_validation_from_row(r: &sqlx::sqlite::SqliteRow) -> CachedValidation
{
	let tm_str: Option<String> = r.get("threshold_metadata");
	let tm = tm_str.and_then(|s| serde_json::from_str(&s).ok());
	(r.get("ce"), r.get("accuracy"), r.get("f1_macro"), r.get("fpr"), tm)
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
	scope: Option<CacheScope<'_>>,
) -> Result<i64>
{
	let now = Utc::now().to_rfc3339();

	let result = sqlx::query(
        r#"INSERT INTO validation_summaries
           (flow_id, experiment_id, validation_point, genome_type, genome_hash, ce, accuracy, f1_macro, fpr, threshold_metadata, cache_key, worker_abi, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(experiment_id, validation_point, genome_type) DO UPDATE SET
             flow_id = excluded.flow_id,
             genome_hash = excluded.genome_hash,
             ce = excluded.ce,
             accuracy = excluded.accuracy,
             f1_macro = excluded.f1_macro,
             fpr = excluded.fpr,
             threshold_metadata = excluded.threshold_metadata,
             cache_key = excluded.cache_key,
             worker_abi = excluded.worker_abi,
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
    .bind(scope.map(|sc| sc.key))
    .bind(scope.map(|sc| sc.worker_abi))
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

#[cfg(test)]
mod cache_scope_tests
{
	//! PAPER-CRITICAL regression cover (10/10/2026): the cross-flow validation
	//! cache must never serve a row across memory modes, trainer ABIs, or from
	//! a legacy (unstamped) row. Keys come from the SAME fixture the Python
	//! builder is tested against (tests/test_validation_cache_key.py).
	use super::*;

	const FIXTURE: &str = include_str!("../../../tests/fixtures/validation_cache_keys.json");

	fn fixture_key(name: &str) -> String
	{
		let v: serde_json::Value = serde_json::from_str(FIXTURE).expect("fixture json");
		let cases = v["cases"].as_array().expect("cases").clone();
		let case = cases.into_iter().find(|c| c["name"] == name).expect("fixture case");
		case["key"].as_str().expect("key").to_string()
	}

	async fn test_pool(tag: &str) -> DbPool
	{
		let path = std::env::temp_dir().join(format!("wnn_valcache_{}_{}.db", tag, std::process::id()));
		let _ = std::fs::remove_file(&path);
		crate::db::init_db(&format!("sqlite://{}?mode=rwc", path.display()))
			.await
			.expect("init_db")
	}

	async fn new_experiment(pool: &DbPool) -> i64
	{
		let flow = sqlx::query("INSERT INTO flows (name) VALUES ('t')").execute(pool).await.unwrap();
		sqlx::query("INSERT INTO experiments (flow_id, name) VALUES (?, 'e')")
			.bind(flow.last_insert_rowid())
			.execute(pool)
			.await
			.unwrap()
			.last_insert_rowid()
	}

	/// One final validation row for genome "g1" in a fresh experiment.
	async fn write_row(pool: &DbPool, scope: Option<CacheScope<'_>>, f1: f64)
	{
		let exp = new_experiment(pool).await;
		upsert_validation_summary(pool, None, exp, "final", "best_f1", "g1", 0.1, 0.9, Some(f1), Some(0.05), None, scope)
			.await
			.unwrap();
	}

	fn scope(key: &str, worker_abi: i64) -> Option<CacheScope<'_>>
	{
		Some(CacheScope { key, worker_abi })
	}

	#[tokio::test]
	async fn quad_flow_never_inherits_a_qsr_row()
	{
		let pool = test_pool("qsr").await;
		let (quad, qsr) = (fixture_key("quad_default"), fixture_key("qsr"));
		write_row(&pool, scope(&qsr, 14), 0.94274).await;
		assert!(get_cached_validation(&pool, "g1", scope(&quad, 14)).await.unwrap().is_none());
		let hit = get_cached_validation(&pool, "g1", scope(&qsr, 14)).await.unwrap();
		assert_eq!(hit.expect("same scope must hit").2, Some(0.94274));
	}

	#[tokio::test]
	async fn equal_scope_rows_resolve_to_the_most_recent()
	{
		let pool = test_pool("tie").await;
		let quad = fixture_key("quad_default");
		write_row(&pool, scope(&quad, 14), 0.80).await;
		write_row(&pool, scope(&quad, 14), 0.85).await;
		let hit = get_cached_validation(&pool, "g1", scope(&quad, 14)).await.unwrap();
		assert_eq!(hit.expect("hit").2, Some(0.85));
	}

	#[tokio::test]
	async fn absent_and_explicit_quad_share_the_cache()
	{
		let pool = test_pool("alias").await;
		let (absent, explicit) = (fixture_key("quad_default"), fixture_key("quad_explicit"));
		write_row(&pool, scope(&absent, 14), 0.9).await;
		assert!(get_cached_validation(&pool, "g1", scope(&explicit, 14)).await.unwrap().is_some());
	}

	#[tokio::test]
	async fn other_trainer_abi_never_hits()
	{
		let pool = test_pool("abi").await;
		let quad = fixture_key("quad_default");
		write_row(&pool, scope(&quad, 12), 0.9).await;
		assert!(get_cached_validation(&pool, "g1", scope(&quad, 14)).await.unwrap().is_none());
	}

	#[tokio::test]
	async fn legacy_unstamped_rows_are_never_served()
	{
		let pool = test_pool("legacy").await;
		write_row(&pool, None, 0.9).await;
		let quad = fixture_key("quad_default");
		assert!(get_cached_validation(&pool, "g1", scope(&quad, 14)).await.unwrap().is_none());
		assert!(get_cached_validation(&pool, "g1", None).await.unwrap().is_none());
	}

	#[tokio::test]
	async fn other_encoding_training_or_classification_never_hits()
	{
		let pool = test_pool("enc").await;
		write_row(&pool, scope(&fixture_key("quad_default"), 14), 0.9).await;
		for name in ["raw", "inv", "oi_off", "feature_top20", "multiclass"]
		{
			let key = fixture_key(name);
			let hit = get_cached_validation(&pool, "g1", scope(&key, 14)).await.unwrap();
			assert!(hit.is_none(), "{name} must not hit a quad_default row");
		}
	}

	#[test]
	fn scope_requires_both_halves()
	{
		assert!(CacheScope::from_parts(Some("k"), Some(14)).is_some());
		assert!(CacheScope::from_parts(Some("k"), None).is_none());
		assert!(CacheScope::from_parts(None, Some(14)).is_none());
		assert!(CacheScope::from_parts(Some(""), Some(14)).is_none());
	}
}
