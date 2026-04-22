use std::fs;
use std::path::{Path, PathBuf};

use serde::Serialize;

use super::config::ExperimentConfig;
use super::result::{ArtifactRef, ExperimentResult};

pub fn prepare_run_dir(
    root_dir: &str,
    mode: &str,
    run_label: &str,
    run_id: &str,
) -> anyhow::Result<PathBuf> {
    let mut p = PathBuf::from(root_dir);
    p.push(mode);
    p.push(sanitize_name(run_label));
    p.push(run_id);
    fs::create_dir_all(&p)?;
    Ok(p)
}

pub fn write_json_file<T: Serialize>(path: &Path, value: &T) -> anyhow::Result<()> {
    let data = serde_json::to_vec_pretty(value)?;
    fs::write(path, data)?;
    Ok(())
}

pub fn snapshot_config(run_dir: &Path, cfg: &ExperimentConfig) -> anyhow::Result<ArtifactRef> {
    let path = run_dir.join("config.snapshot.json");
    write_json_file(&path, cfg)?;
    Ok(ArtifactRef {
        name: "config_snapshot".to_string(),
        path: path.to_string_lossy().to_string(),
        kind: "json".to_string(),
    })
}

pub fn snapshot_result(run_dir: &Path, result: &ExperimentResult) -> anyhow::Result<ArtifactRef> {
    let path = run_dir.join("result.json");
    write_json_file(&path, result)?;
    Ok(ArtifactRef {
        name: "result".to_string(),
        path: path.to_string_lossy().to_string(),
        kind: "json".to_string(),
    })
}

fn sanitize_name(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' {
            out.push(ch);
        } else {
            out.push('_');
        }
    }
    while out.contains("__") {
        out = out.replace("__", "_");
    }
    out.trim_matches('_').to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::experiments::config::{
        DataConfig, DetectorConfig, DetectorType, EvaluationConfig, ExperimentConfig,
        ExperimentMode, FeatureConfig, FeatureFamilyConfig, ModelConfig, OutputConfig,
        ReproducibilityConfig, RunMetaConfig, ScalingPolicyConfig, TrainingMode,
    };

    fn minimal_cfg() -> ExperimentConfig {
        ExperimentConfig {
            meta: RunMetaConfig {
                run_label: "abc".to_string(),
                notes: None,
            },
            mode: ExperimentMode::Synthetic,
            data: DataConfig::Synthetic {
                scenario_id: "s".to_string(),
                horizon: 20,
                dataset_id: None,
            },
            features: FeatureConfig {
                family: FeatureFamilyConfig::LogReturn,
                scaling: ScalingPolicyConfig::None,
                session_aware: false,
            },
            model: ModelConfig {
                k_regimes: 2,
                training: TrainingMode::FitOffline,
                em_max_iter: 10,
                em_tol: 1e-5,
            },
            detector: DetectorConfig {
                detector_type: DetectorType::Surprise,
                threshold: 2.0,
                persistence_required: 1,
                cooldown: 0,
            },
            evaluation: EvaluationConfig::Synthetic { matching_window: 5 },
            output: OutputConfig {
                root_dir: std::env::temp_dir().to_string_lossy().to_string(),
                write_json: true,
                write_csv: false,
                save_traces: false,
            },
            reproducibility: ReproducibilityConfig {
                seed: Some(1),
                deterministic_run_id: true,
                save_config_snapshot: true,
                record_git_info: false,
            },
        }
    }

    #[test]
    fn output_dir_is_created() {
        let base = std::env::temp_dir();
        let root = base.join("proteus_exp_artifact_test");
        let d = prepare_run_dir(
            root.to_string_lossy().as_ref(),
            "synthetic",
            "run one",
            "rid",
        )
        .unwrap();
        assert!(d.exists());
    }

    #[test]
    fn config_snapshot_written() {
        let cfg = minimal_cfg();
        let run_dir =
            prepare_run_dir(cfg.output.root_dir.as_str(), "synthetic", "cfgsnap", "rid2").unwrap();
        let a = snapshot_config(&run_dir, &cfg).unwrap();
        assert!(std::path::Path::new(&a.path).exists());
    }
}
