//! Log file parser for phased search experiments
//!
//! Parses log lines like:
//! - `21:10:12 | [ArchitectureTS] Iter 001/250: best_harmonic=(CE=10.5401, Acc=8.2007%), ...`
//! - `21:03:32 | Phase 3a: GA Connections Only (Best Harmonic)    10.4185      33472.8     0.97%`
//! - `20:45:23 |   Phase 2a: GA Bits Only`

use chrono::NaiveTime;
use regex::Regex;
use std::sync::LazyLock;

use crate::models::*;

// Regex patterns compiled once
static TIMESTAMP_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^(\d{2}:\d{2}:\d{2}) \| (.*)$").unwrap());

// Phase start lines - two formats:
// Old: "Phase 1a: GA Neurons Only" or "Phase 1b: TS Neurons Only (refine)"
// New: "1-GA-Neurons" or "2-TS-Bits"
static PHASE_START_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^Phase (\d+[ab]): (GA|TS) (\w+) Only(?: \(refine\))?$").unwrap());

// New format: "1-GA-Neurons", "2-TS-Bits", etc.
static PHASE_START_NEW_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^(\d+)-(GA|TS)-(\w+)$").unwrap());

static GA_ITER_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"\[(?:Architecture)?GA\] Gen (\d+)/(\d+): best=([0-9.]+), avg=([0-9.]+)(?: \(subset\))? \(([0-9.]+)s\)",
    )
    .unwrap()
});

// New format: "[Gen 017/250] Genome 01/40 (New ): CE=10.6133, Acc=0.0240%"
// or "[Gen 017/250] Genome 01/10 (Elite): CE=10.5123, Acc=0.5400%"
static GENOME_PROGRESS_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"\[Gen (\d+)/(\d+)\] Genome (\d+)/(\d+) \((Elite|New\s*)\): CE=([0-9.]+), Acc=([0-9.]+)%",
    )
    .unwrap()
});

static TS_ITER_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"\[ArchitectureTS\] Iter (\d+)/(\d+): best_harmonic=\(CE=([0-9.]+), Acc=([0-9.]+)%\), best_ce=([0-9.]+), best_acc=([0-9.]+)%(?: \(subset\))? \(([0-9.]+)s\)",
    )
    .unwrap()
});

static PHASE_RESULT_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"Phase \d+[ab]: .+?\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)%\s+([+-][0-9.]+)%",
    )
    .unwrap()
});

static HEALTH_CHECK_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"Top-(\d+) mean \(full\): CE=([0-9.]+), Acc=([0-9.]+)%").unwrap()
});

// Phase summary table row: "Phase 1a: GA Neurons Only │ top-10 mean │ 10.4041 │ 32993.5 │ 0.10%"
// or continuation:         "                         │ best CE     │ 10.4007 │ 32882.9 │ 0.11%"
static PHASE_SUMMARY_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"^(Phase \d+[ab]: .+?|Baseline|\s+)\s*│\s*(top-\d+ mean|best CE|best Acc)\s*│\s*([0-9.]+)\s*│\s*([0-9.]+)\s*│\s*([0-9.]+)%").unwrap()
});

// Final results marker
static FINAL_RESULTS_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"FINAL RESULTS").unwrap()
});

/// Parsed log event
#[derive(Debug)]
pub enum LogEvent {
    PhaseStart { name: String, phase_type: PhaseType },
    GaIteration { generation: i32, total: i32, best: f64, avg: f64, elapsed: f64 },
    TsIteration { iter: i32, total: i32, best_harmonic_ce: f64, best_harmonic_acc: f64, best_ce: f64, best_acc: f64, elapsed: f64 },
    /// Per-genome progress from Rust-side logging
    GenomeProgress { generation: i32, total_gens: i32, genome_idx: i32, total_genomes: i32, is_elite: bool, ce: f64, accuracy: f64 },
    PhaseResult { ce: f64, memory: f64, accuracy: f64, improvement: f64 },
    HealthCheck { k: i32, ce: f64, accuracy: f64 },
    /// Final results marker - indicates summary table follows
    FinalResults,
    /// Phase summary row from comparison table
    PhaseSummaryRow { phase_name: String, metric_type: String, ce: f64, ppl: f64, accuracy: f64 },
    Unknown(String),
}

/// Parse a single log line
pub fn parse_line(line: &str) -> Option<(NaiveTime, LogEvent)> {
    let caps = TIMESTAMP_RE.captures(line)?;
    let time = NaiveTime::parse_from_str(caps.get(1)?.as_str(), "%H:%M:%S").ok()?;
    let content = caps.get(2)?.as_str().trim();

    let event = if let Some(caps) = PHASE_START_RE.captures(content) {
        // Old format: "Phase 1a: GA Neurons Only"
        let name = format!("Phase {}: {} {}",
            caps.get(1)?.as_str(),
            caps.get(2)?.as_str(),
            caps.get(3)?.as_str()
        );
        let phase_type = match (caps.get(2)?.as_str(), caps.get(3)?.as_str().to_lowercase().as_str()) {
            ("GA", "neurons") => PhaseType::GaNeurons,
            ("TS", "neurons") => PhaseType::TsNeurons,
            ("GA", "bits") => PhaseType::GaBits,
            ("TS", "bits") => PhaseType::TsBits,
            ("GA", "connections") => PhaseType::GaConnections,
            ("TS", "connections") => PhaseType::TsConnections,
            _ => return Some((time, LogEvent::Unknown(content.to_string()))),
        };
        LogEvent::PhaseStart { name, phase_type }
    } else if let Some(caps) = PHASE_START_NEW_RE.captures(content) {
        // New format: "1-GA-Neurons", "2-TS-Bits", etc.
        let phase_num = caps.get(1)?.as_str();
        let algo = caps.get(2)?.as_str();
        let target = caps.get(3)?.as_str();
        let name = format!("{}-{}-{}", phase_num, algo, target);
        let phase_type = match (algo, target.to_lowercase().as_str()) {
            ("GA", "neurons") => PhaseType::GaNeurons,
            ("TS", "neurons") => PhaseType::TsNeurons,
            ("GA", "bits") => PhaseType::GaBits,
            ("TS", "bits") => PhaseType::TsBits,
            ("GA", "connections") => PhaseType::GaConnections,
            ("TS", "connections") => PhaseType::TsConnections,
            _ => return Some((time, LogEvent::Unknown(content.to_string()))),
        };
        LogEvent::PhaseStart { name, phase_type }
    } else if let Some(caps) = GA_ITER_RE.captures(content) {
        LogEvent::GaIteration {
            generation: caps.get(1)?.as_str().parse().ok()?,
            total: caps.get(2)?.as_str().parse().ok()?,
            best: caps.get(3)?.as_str().parse().ok()?,
            avg: caps.get(4)?.as_str().parse().ok()?,
            elapsed: caps.get(5)?.as_str().parse().ok()?,
        }
    } else if let Some(caps) = GENOME_PROGRESS_RE.captures(content) {
        // New format: "[Gen 017/250] Genome 01/40 (New ): CE=10.6133, Acc=0.0240%"
        let genome_type = caps.get(5)?.as_str().trim();
        LogEvent::GenomeProgress {
            generation: caps.get(1)?.as_str().parse().ok()?,
            total_gens: caps.get(2)?.as_str().parse().ok()?,
            genome_idx: caps.get(3)?.as_str().parse().ok()?,
            total_genomes: caps.get(4)?.as_str().parse().ok()?,
            is_elite: genome_type == "Elite",
            ce: caps.get(6)?.as_str().parse().ok()?,
            accuracy: caps.get(7)?.as_str().parse().ok()?,
        }
    } else if let Some(caps) = TS_ITER_RE.captures(content) {
        LogEvent::TsIteration {
            iter: caps.get(1)?.as_str().parse().ok()?,
            total: caps.get(2)?.as_str().parse().ok()?,
            best_harmonic_ce: caps.get(3)?.as_str().parse().ok()?,
            best_harmonic_acc: caps.get(4)?.as_str().parse().ok()?,
            best_ce: caps.get(5)?.as_str().parse().ok()?,
            best_acc: caps.get(6)?.as_str().parse().ok()?,
            elapsed: caps.get(7)?.as_str().parse().ok()?,
        }
    } else if let Some(caps) = PHASE_RESULT_RE.captures(content) {
        LogEvent::PhaseResult {
            ce: caps.get(1)?.as_str().parse().ok()?,
            memory: caps.get(2)?.as_str().parse().ok()?,
            accuracy: caps.get(3)?.as_str().parse().ok()?,
            improvement: caps.get(4)?.as_str().parse().ok()?,
        }
    } else if let Some(caps) = HEALTH_CHECK_RE.captures(content) {
        LogEvent::HealthCheck {
            k: caps.get(1)?.as_str().parse().ok()?,
            ce: caps.get(2)?.as_str().parse().ok()?,
            accuracy: caps.get(3)?.as_str().parse().ok()?,
        }
    } else if FINAL_RESULTS_RE.is_match(content) {
        LogEvent::FinalResults
    } else if let Some(caps) = PHASE_SUMMARY_RE.captures(content) {
        let phase_name = caps.get(1)?.as_str().trim().to_string();
        LogEvent::PhaseSummaryRow {
            phase_name: if phase_name.is_empty() { String::new() } else { phase_name },
            metric_type: caps.get(2)?.as_str().to_string(),
            ce: caps.get(3)?.as_str().parse().ok()?,
            ppl: caps.get(4)?.as_str().parse().ok()?,
            accuracy: caps.get(5)?.as_str().parse().ok()?,
        }
    } else {
        LogEvent::Unknown(content.to_string())
    };

    Some((time, event))
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Timelike;

    #[test]
    fn test_parse_ts_iteration() {
        let line = "21:10:12 | [ArchitectureTS] Iter 001/250: best_harmonic=(CE=10.5401, Acc=8.2007%), best_ce=10.4767, best_acc=8.2007% (subset) (96.5s)";
        let (time, event) = parse_line(line).unwrap();
        assert_eq!(time.hour(), 21);
        match event {
            LogEvent::TsIteration { iter, best_harmonic_ce, elapsed, .. } => {
                assert_eq!(iter, 1);
                assert!((best_harmonic_ce - 10.5401).abs() < 0.001);
                assert!((elapsed - 96.5).abs() < 0.1);
            }
            _ => panic!("Expected TsIteration"),
        }
    }

    #[test]
    fn test_parse_phase_start() {
        let line = "20:45:23 |   Phase 2a: GA Bits Only";
        let (_, event) = parse_line(line).unwrap();
        match event {
            LogEvent::PhaseStart { phase_type, .. } => {
                assert_eq!(phase_type, PhaseType::GaBits);
            }
            _ => panic!("Expected PhaseStart"),
        }
    }

    #[test]
    fn test_parse_phase_start_new_format() {
        let line = "20:45:23 | 1-GA-Neurons";
        let (_, event) = parse_line(line).unwrap();
        match event {
            LogEvent::PhaseStart { name, phase_type } => {
                assert_eq!(name, "1-GA-Neurons");
                assert_eq!(phase_type, PhaseType::GaNeurons);
            }
            _ => panic!("Expected PhaseStart"),
        }

        // Test another new format variant
        let line2 = "21:10:05 | 2-TS-Bits";
        let (_, event2) = parse_line(line2).unwrap();
        match event2 {
            LogEvent::PhaseStart { name, phase_type } => {
                assert_eq!(name, "2-TS-Bits");
                assert_eq!(phase_type, PhaseType::TsBits);
            }
            _ => panic!("Expected PhaseStart"),
        }
    }

    #[test]
    fn test_parse_genome_progress() {
        // Test new genome progress format (offspring)
        let line = "00:50:36 | [Gen 017/250] Genome 01/40 (New ): CE=10.6133, Acc=0.0240%";
        let (_, event) = parse_line(line).unwrap();
        match event {
            LogEvent::GenomeProgress { generation, total_gens, genome_idx, total_genomes, is_elite, ce, accuracy } => {
                assert_eq!(generation, 17);
                assert_eq!(total_gens, 250);
                assert_eq!(genome_idx, 1);
                assert_eq!(total_genomes, 40);
                assert!(!is_elite);
                assert!((ce - 10.6133).abs() < 0.001);
                assert!((accuracy - 0.0240).abs() < 0.001);
            }
            _ => panic!("Expected GenomeProgress"),
        }

        // Test elite genome format
        let line2 = "00:48:56 | [Gen 030/250] Genome 05/10 (Elite): CE=10.5401, Acc=0.5400%";
        let (_, event2) = parse_line(line2).unwrap();
        match event2 {
            LogEvent::GenomeProgress { generation, genome_idx, is_elite, ce, accuracy, .. } => {
                assert_eq!(generation, 30);
                assert_eq!(genome_idx, 5);
                assert!(is_elite);
                assert!((ce - 10.5401).abs() < 0.001);
                assert!((accuracy - 0.5400).abs() < 0.001);
            }
            _ => panic!("Expected GenomeProgress"),
        }
    }
}
