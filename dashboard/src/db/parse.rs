//! Shared row/status/datetime parse helpers (split from db/mod.rs `queries`).

use super::*;

pub(crate) fn parse_gating_status(s: &str) -> GatingStatus
{
	match s
	{
		"pending" => GatingStatus::Pending,
		"running" => GatingStatus::Running,
		"completed" => GatingStatus::Completed,
		"failed" => GatingStatus::Failed,
		_ => GatingStatus::Pending,
	}
}

pub(crate) fn parse_experiment_status(s: &str) -> ExperimentStatus
{
	match s
	{
		"pending" => ExperimentStatus::Pending,
		"queued" => ExperimentStatus::Queued,
		"running" => ExperimentStatus::Running,
		"paused" => ExperimentStatus::Paused,
		"completed" => ExperimentStatus::Completed,
		"failed" => ExperimentStatus::Failed,
		"cancelled" => ExperimentStatus::Cancelled,
		_ => ExperimentStatus::Pending,
	}
}

pub(crate) fn parse_fitness_calculator(s: &str) -> FitnessCalculator
{
	match s
	{
		"ce" => FitnessCalculator::Ce,
		"harmonic_rank" => FitnessCalculator::HarmonicRank,
		"weighted_harmonic" => FitnessCalculator::WeightedHarmonic,
		"normalized" => FitnessCalculator::Normalized,
		"normalized_harmonic" => FitnessCalculator::NormalizedHarmonic,
		"ids_security" => FitnessCalculator::IdsSecurity,
		"ids_recall" => FitnessCalculator::IdsRecall,
		_ => FitnessCalculator::HarmonicRank,
	}
}

// =============================================================================
// Helper functions
// =============================================================================

pub(crate) fn parse_datetime(s: String) -> Result<DateTime<Utc>>
{
	// Try RFC 3339 first (standard format: 2026-02-02T05:48:49Z or 2026-02-02T05:48:49+00:00)
	if let Ok(dt) = DateTime::parse_from_rfc3339(&s)
	{
		return Ok(dt.with_timezone(&Utc));
	}

	// Try ISO 8601 with space instead of T (legacy format: 2026-02-02 05:48:49)
	// Also handles dates with/without microseconds
	use chrono::NaiveDateTime;
	let formats = [
		"%Y-%m-%d %H:%M:%S%.f", // With optional fractional seconds
		"%Y-%m-%d %H:%M:%S",    // Without fractional seconds
		"%Y-%m-%dT%H:%M:%S%.f", // T separator with fractional (no timezone)
		"%Y-%m-%dT%H:%M:%S",    // T separator without fractional (no timezone)
	];

	for fmt in formats
	{
		if let Ok(naive) = NaiveDateTime::parse_from_str(&s, fmt)
		{
			return Ok(naive.and_utc());
		}
	}

	// If all parsing fails, return an error with context
	Err(anyhow::anyhow!("Failed to parse datetime: '{}'", s))
}

pub(crate) fn parse_architecture_type(s: &str) -> ArchitectureType
{
	match s
	{
		"bitwise" => ArchitectureType::Bitwise,
		"multi_stage" | "multistage" => ArchitectureType::MultiStage,
		"ids" => ArchitectureType::Ids,
		"controller" => ArchitectureType::Controller,
		_ => ArchitectureType::Tiered,
	}
}

pub(crate) fn parse_flow_status(s: &str) -> FlowStatus
{
	match s
	{
		"pending" => FlowStatus::Pending,
		"queued" => FlowStatus::Queued,
		"running" => FlowStatus::Running,
		"paused" => FlowStatus::Paused,
		"completed" => FlowStatus::Completed,
		"failed" => FlowStatus::Failed,
		"cancelled" => FlowStatus::Cancelled,
		_ => FlowStatus::Pending,
	}
}

/// Compute max_iterations from phase_type and flow config.
/// Returns None if phase_type is unknown or config doesn't have the needed params.
pub(crate) fn compute_max_iterations_from_phase_type(
	phase_type: Option<&str>,
	config: &crate::models::FlowConfig,
) -> Option<i32>
{
	let pt = phase_type?;
	if pt == "grid_search"
	{
		return Some(1);
	}
	if pt.starts_with("neurogenesis")
		|| pt.starts_with("synaptogenesis")
		|| pt.starts_with("axonogenesis")
	{
		return config
			.params
			.get("adaptation_iterations")
			.and_then(|v| v.as_i64())
			.map(|v| v as i32);
	}
	if pt.starts_with("ga")
	{
		return config
			.params
			.get("ga_generations")
			.and_then(|v| v.as_i64())
			.map(|v| v as i32);
	}
	if pt.starts_with("ts")
	{
		return config
			.params
			.get("ts_iterations")
			.and_then(|v| v.as_i64())
			.map(|v| v as i32);
	}
	None
}
