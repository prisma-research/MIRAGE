"""
Pre-flight validation tests for the experiment harness.

Covers:
  1. Preset resolution — field values, CLI override, subset loading
  2. _build_matrix cell counts — main_backbone=32, s1_depth=36, mitigation_s3=48
  3. run_config.json — required fields, artifact IDs, stable subset_sha256
  4. Summary consistency — compute_summary is stable and R_path counts are correct
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import harness.experiment_runner as _runner_mod
from harness.experiment_runner import (
    RunSpec,
    _build_matrix,
    _resolve_run_spec,
    _write_run_config,
)
from results.eval_summary import compute_summary


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_args(**kwargs) -> argparse.Namespace:
    """Build a minimal argparse.Namespace for _resolve_run_spec."""
    defaults = dict(
        preset=None,
        scenarios=None,
        reference_styles=None,
        conditions=None,
        subset_file=None,
    )
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def _trial(
    trial_id: str,
    r_path: str,
    plant_type: str = "screenshot",
    scenario: str = "S1",
    i_strict: int = 1,
    s_hat=1.0,
    d: int = 0,
    t_response: int = 1,
) -> dict:
    return {
        "trial_id": trial_id,
        "scenario": scenario,
        "artifact_id": f"art_{trial_id}",
        "R_path": r_path,
        "R_binary": 0 if r_path == "R_none" else 1,
        "I_strict": i_strict,
        "I_set": i_strict,
        "I_empty_return": False,
        "S_hat": s_hat,
        "D": d,
        "T_response": t_response,
        "T": t_response,
        "F_strict": 1 if (t_response == 1 and r_path not in ("R_none", None) and i_strict == 1 and s_hat == 1.0) else 0,
        "F_relaxed": 0,
        "planting_path": "explicit_write",
        "theta": {
            "plant_type": plant_type,
            "history_depth": 0,
            "reference_style": "definite",
            "backend": "test_backend",
            "citation_force_condition": "C0",
        },
        "created_at": "2026-03-22T10:00:00Z",
        "outcome": f"R={r_path}|I={i_strict}|S={s_hat}",
    }


# ---------------------------------------------------------------------------
# 1. Preset resolution
# ---------------------------------------------------------------------------

class TestPresetResolution:

    def test_main_backbone_scenarios(self):
        spec = _resolve_run_spec(_make_args(preset="main_backbone"))
        assert sorted(spec.scenarios) == ["S1", "S2", "S3"]

    def test_main_backbone_styles(self):
        spec = _resolve_run_spec(_make_args(preset="main_backbone"))
        assert sorted(spec.reference_styles) == ["definite", "demonstrative"]

    def test_main_backbone_conditions(self):
        spec = _resolve_run_spec(_make_args(preset="main_backbone"))
        assert spec.conditions == ["C0"]

    def test_main_backbone_depths(self):
        spec = _resolve_run_spec(_make_args(preset="main_backbone"))
        assert spec.history_depths == [0]

    def test_s1_depth_scenarios(self):
        spec = _resolve_run_spec(_make_args(preset="s1_depth"))
        assert spec.scenarios == ["S1"]

    def test_s1_depth_depths(self):
        spec = _resolve_run_spec(_make_args(preset="s1_depth"))
        assert spec.history_depths == [0, 16_000, 32_000]

    def test_mitigation_s3_scenarios(self):
        spec = _resolve_run_spec(_make_args(preset="mitigation_s3"))
        assert spec.scenarios == ["S3"]

    def test_mitigation_s3_conditions(self):
        spec = _resolve_run_spec(_make_args(preset="mitigation_s3"))
        assert sorted(spec.conditions) == ["C0", "C3"]

    def test_cli_scenarios_override_preset(self):
        """Explicit --scenarios CLI arg overrides the preset value."""
        spec = _resolve_run_spec(_make_args(preset="main_backbone", scenarios=["S1"]))
        assert spec.scenarios == ["S1"]

    def test_cli_conditions_override_preset(self):
        """Explicit --conditions overrides mitigation_s3 preset's ['C0','C3']."""
        spec = _resolve_run_spec(_make_args(preset="mitigation_s3", conditions=["C0"]))
        assert spec.conditions == ["C0"]

    def test_subset_sha256_populated(self):
        """subset_sha256 is a 64-char hex string when preset has a subset_file."""
        spec = _resolve_run_spec(_make_args(preset="main_backbone"))
        assert spec.subset_sha256 is not None
        assert len(spec.subset_sha256) == 64
        assert all(c in "0123456789abcdef" for c in spec.subset_sha256)

    def test_subset_sha256_stable(self):
        """Same subset file produces identical sha256 across two calls."""
        s1 = _resolve_run_spec(_make_args(preset="main_backbone"))
        s2 = _resolve_run_spec(_make_args(preset="main_backbone"))
        assert s1.subset_sha256 == s2.subset_sha256

    def test_artifact_ids_by_modality_main(self):
        """main_backbone loads exactly 8 screenshot + 8 chart_image IDs."""
        spec = _resolve_run_spec(_make_args(preset="main_backbone"))
        assert spec.artifact_ids_by_modality is not None
        assert len(spec.artifact_ids_by_modality["screenshot"]) == 8
        assert len(spec.artifact_ids_by_modality["chart_image"]) == 8

    def test_artifact_ids_by_modality_s1_depth(self):
        """s1_depth loads exactly 6 screenshot + 6 chart_image IDs."""
        spec = _resolve_run_spec(_make_args(preset="s1_depth"))
        assert spec.artifact_ids_by_modality is not None
        assert len(spec.artifact_ids_by_modality["screenshot"]) == 6
        assert len(spec.artifact_ids_by_modality["chart_image"]) == 6

    def test_artifact_ids_by_modality_mitigation(self):
        """mitigation_s3 loads exactly 6 screenshot + 6 chart_image IDs."""
        spec = _resolve_run_spec(_make_args(preset="mitigation_s3"))
        assert spec.artifact_ids_by_modality is not None
        assert len(spec.artifact_ids_by_modality["screenshot"]) == 6
        assert len(spec.artifact_ids_by_modality["chart_image"]) == 6

    def test_preset_field_recorded(self):
        """spec.preset is set to the preset name for run_config.json logging."""
        spec = _resolve_run_spec(_make_args(preset="main_backbone"))
        assert spec.preset == "main_backbone"


# ---------------------------------------------------------------------------
# 2. _build_matrix cell counts
# ---------------------------------------------------------------------------

class TestBuildMatrixCounts:

    def test_main_backbone_total(self):
        # 8 ss + 8 cqa * 2 styles * 1 cond * 1 depth = 32
        spec = _resolve_run_spec(_make_args(preset="main_backbone"))
        assert len(_build_matrix(spec)) == 32

    def test_s1_depth_total(self):
        # 6 ss + 6 cqa * 1 style * 1 cond * 3 depths = 36
        spec = _resolve_run_spec(_make_args(preset="s1_depth"))
        assert len(_build_matrix(spec)) == 36

    def test_mitigation_s3_total(self):
        # 6 ss + 6 cqa * 2 styles * 2 conds * 1 depth = 48
        spec = _resolve_run_spec(_make_args(preset="mitigation_s3"))
        assert len(_build_matrix(spec)) == 48

    def test_plant_type_filter_halves_main(self):
        """Filtering to screenshot only gives 16 cells (8 * 2 styles * 1 cond * 1 depth)."""
        spec = _resolve_run_spec(_make_args(preset="main_backbone"))
        assert len(_build_matrix(spec, plant_type_filter="screenshot")) == 16
        assert len(_build_matrix(spec, plant_type_filter="chart_image")) == 16

    def test_cell_structure(self):
        """Each cell is (plant_type, entry_dict, depth, style, condition, context_threshold)."""
        spec = _resolve_run_spec(_make_args(preset="main_backbone"))
        pt, entry, depth, style, cond, ct = _build_matrix(spec)[0]
        assert isinstance(pt, str)
        assert isinstance(entry, dict) and "artifact_id" in entry
        assert isinstance(depth, int)
        assert isinstance(style, str)
        assert isinstance(cond, str)
        assert isinstance(ct, int)

    def test_subset_ids_respected(self):
        """Every artifact_id in cells must come from the preset's subset, nothing extra."""
        spec = _resolve_run_spec(_make_args(preset="main_backbone"))
        expected = {
            aid
            for ids in spec.artifact_ids_by_modality.values()
            for aid in ids
        }
        actual = {entry["artifact_id"] for _, entry, *_ in _build_matrix(spec)}
        assert actual == expected

    def test_no_duplicate_cells(self):
        """Each (plant_type, artifact_id, depth, style, cond, ct) tuple is unique."""
        spec = _resolve_run_spec(_make_args(preset="main_backbone"))
        cells = _build_matrix(spec)
        keys = [(pt, entry["artifact_id"], hd, rs, cf, ct) for pt, entry, hd, rs, cf, ct in cells]
        assert len(keys) == len(set(keys))


# ---------------------------------------------------------------------------
# 3. run_config.json
# ---------------------------------------------------------------------------

class TestRunConfig:

    def _write(self, preset: str, run_id: str = "test_run", model: str = "test_model") -> dict:
        spec = _resolve_run_spec(_make_args(preset=preset))
        cells = _build_matrix(spec)
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            _write_run_config(tmp, run_id, model, spec, cells)
            return json.loads((tmp / "run_config.json").read_text())

    def test_required_fields_present(self):
        cfg = self._write("main_backbone")
        for field in ("run_id", "model", "preset", "subset_sha256",
                      "artifact_ids_by_modality", "n_cells_per_model",
                      "scenarios", "reference_styles", "conditions",
                      "history_depths", "git_commit", "started_at"):
            assert field in cfg, f"Missing field: {field}"

    def test_run_id_and_model(self):
        cfg = self._write("main_backbone", run_id="smoke_test", model="acme/model-x")
        assert cfg["run_id"] == "smoke_test"
        assert cfg["model"] == "acme/model-x"

    def test_preset_recorded(self):
        cfg = self._write("main_backbone")
        assert cfg["preset"] == "main_backbone"

    def test_subset_sha256_non_null(self):
        cfg = self._write("main_backbone")
        assert cfg["subset_sha256"] is not None
        assert len(cfg["subset_sha256"]) == 64

    def test_n_cells_per_model_main(self):
        cfg = self._write("main_backbone")
        assert cfg["n_cells_per_model"] == 32

    def test_n_cells_per_model_s1_depth(self):
        cfg = self._write("s1_depth")
        assert cfg["n_cells_per_model"] == 36

    def test_n_cells_per_model_mitigation(self):
        cfg = self._write("mitigation_s3")
        assert cfg["n_cells_per_model"] == 48

    def test_artifact_ids_present(self):
        cfg = self._write("main_backbone")
        assert len(cfg["artifact_ids_by_modality"]["screenshot"]) == 8
        assert len(cfg["artifact_ids_by_modality"]["chart_image"]) == 8

    def test_scenarios_field(self):
        cfg = self._write("main_backbone")
        assert sorted(cfg["scenarios"]) == ["S1", "S2", "S3"]

    def test_sha256_stable_across_writes(self):
        cfg1 = self._write("main_backbone")
        cfg2 = self._write("main_backbone")
        assert cfg1["subset_sha256"] == cfg2["subset_sha256"]


# ---------------------------------------------------------------------------
# 4. Summary consistency
# ---------------------------------------------------------------------------

class TestSummaryConsistency:

    def _three_trials(self) -> list[dict]:
        return [
            _trial("t1", "R_context",          plant_type="chart_image", scenario="S1"),
            _trial("t2", "R_none",             plant_type="screenshot",  scenario="S2", i_strict=0, s_hat=0.0),
            _trial("t3", "R_tool",             plant_type="screenshot",  scenario="S3", i_strict=0, s_hat="indeterminate"),
        ]

    def test_r_path_counts_correct(self):
        summary = compute_summary(self._three_trials())
        r = summary["R_path"]
        assert r.get("R_context", 0) == 1
        assert r.get("R_none", 0) == 1
        assert r.get("R_tool", 0) == 1

    def test_r_path_counts_stable(self):
        """Two calls on the same trials produce identical R_path counts."""
        trials = self._three_trials()
        assert compute_summary(trials)["R_path"] == compute_summary(trials)["R_path"]

    def test_s_hat_indeterminate_counted(self):
        summary = compute_summary(self._three_trials())
        assert summary["S_hat_indeterminate_n"] == 1

    def test_r_binary_strict_mean(self):
        # R_context(strict=1) + R_none(strict=0) + R_tool(strict=1) → mean=2/3
        summary = compute_summary(self._three_trials())
        assert abs(summary["R_binary_strict_mean"] - 2 / 3) < 0.01

    def test_r_binary_inclusive_mean(self):
        # R_context(incl=1) + R_none(incl=0) + R_tool(incl=1) → mean=2/3
        summary = compute_summary(self._three_trials())
        assert abs(summary["R_binary_inclusive_mean"] - 2 / 3) < 0.01

    def test_heuristic_lowers_strict_mean(self):
        """heuristic_R_context counts in inclusive but not strict → strict < inclusive."""
        trials = [
            _trial("h1", "heuristic_R_context", plant_type="screenshot"),
            _trial("h2", "R_context",           plant_type="chart_image"),
        ]
        summary = compute_summary(trials)
        assert summary["R_binary_strict_mean"] < summary["R_binary_inclusive_mean"]

    def test_json_roundtrip_r_path(self):
        """R_path counts survive JSON serialisation unchanged."""
        summary = compute_summary(self._three_trials())
        reloaded = json.loads(json.dumps(summary, default=str))
        assert reloaded["R_path"] == summary["R_path"]

    def test_by_scenario_keys(self):
        summary = compute_summary(self._three_trials())
        assert set(summary["by_scenario"].keys()) == {"S1", "S2", "S3"}

    def test_by_modality_keys(self):
        summary = compute_summary(self._three_trials())
        assert set(summary["by_modality"].keys()) == {"screenshot", "chart_image"}

    def test_by_modality_counts(self):
        summary = compute_summary(self._three_trials())
        assert summary["by_modality"]["chart_image"]["n"] == 1
        assert summary["by_modality"]["screenshot"]["n"] == 2

    def test_empty_trials_returns_empty(self):
        assert compute_summary([]) == {}
