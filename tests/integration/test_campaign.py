from __future__ import annotations

import json
from pathlib import Path

from fr_gvi.experiments import campaign


def test_config_parsing_and_seed_reproducibility(tmp_path: Path) -> None:
    config = {
        "id": "tiny",
        "experiment": "A",
        "tier": "smoke",
        "master_seed": 9,
        "target": {"kind": "gaussian", "dimension": 2, "condition": 2.0},
        "iterations": 1,
        "methods": [{"name": "FR--KL", "step_size": 0.1}],
    }
    path = tmp_path / "tiny.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    loaded = campaign.load_configs([str(path)])
    assert loaded[0][1] == config
    assert campaign.seed_for(9, 3, 2) == campaign.seed_for(9, 3, 2)
    assert campaign.seed_for(9, 3, 2) != campaign.seed_for(9, 3, 3)


def test_campaign_resume_skips_matching_completed_job(tmp_path: Path, monkeypatch: object) -> None:
    results = tmp_path / "results"
    monkeypatch.setattr(campaign, "RESULTS", results)
    monkeypatch.setattr(campaign, "STATE_PATH", results / "manifests" / "campaign_state.json")
    config = {
        "id": "resume_tiny",
        "experiment": "A",
        "tier": "smoke",
        "master_seed": 17,
        "target": {"kind": "gaussian", "dimension": 2, "condition": 2.0, "mean_scale": 0.0},
        "iterations": 1,
        "methods": [{"name": "FR--KL", "step_size": 0.1}],
    }
    config_path = tmp_path / "resume.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    state = {"version": 1, "runs": {}}
    first = campaign.run_config(
        config_path,
        config,
        state=state,
        force=False,
        deadline=float("inf"),
    )
    second = campaign.run_config(
        config_path,
        config,
        state=state,
        force=False,
        deadline=float("inf"),
    )
    assert first["completed"] == 1
    assert second["skipped"] == 1
    assert list((results / "raw").rglob("*.csv"))


def test_all_smoke_configs_parse() -> None:
    configs = campaign.load_configs([str(campaign.ROOT / "configs" / "smoke")])
    assert {str(config["experiment"]) for _, config in configs} == set("ABCDEFGHIJKL")

