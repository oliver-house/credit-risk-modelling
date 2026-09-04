import pytest

from update_readme import (
    END_MARKER,
    START_MARKER,
    render_provenance,
    render_table,
    splice,
)

RESULTS = {
    "lgbm_oof_auc": 0.78791,
    "xgb_oof_auc": 0.79102,
    "catboost_oof_auc": 0.79087,
    "ensemble_oof_auc": 0.79348,
    "best_weights": {"lgbm": 0.14, "xgb": 0.42, "catboost": 0.44},
}


def test_render_table_matches_results_payload():
    table = render_table(RESULTS)
    assert "| LightGBM | 0.78791 | 0.14 |" in table
    assert "| XGBoost | 0.79102 | 0.42 |" in table
    assert "| CatBoost | 0.79087 | 0.44 |" in table
    assert "| **Ensemble** | **0.79348** | — |" in table


def test_render_table_pads_auc_to_five_decimals():
    table = render_table({**RESULTS, "lgbm_oof_auc": 0.7})
    assert "| LightGBM | 0.70000 |" in table


def test_splice_replaces_only_the_marked_region():
    readme = f"# Title\n\nintro\n\n{START_MARKER}\n\nold table\n\n{END_MARKER}\n\noutro\n"
    out = splice(readme, "new table")

    assert "new table" in out
    assert "old table" not in out
    assert out.startswith("# Title\n\nintro")
    assert out.endswith("outro\n")


def test_splice_is_idempotent():
    readme = f"{START_MARKER}\n\nold\n\n{END_MARKER}\n"
    once = splice(readme, "table")
    assert splice(once, "table") == once


def test_splice_rejects_a_readme_without_markers():
    with pytest.raises(ValueError, match="missing"):
        splice("# Title\n\nno markers here\n", "table")


def test_splice_rejects_markers_in_the_wrong_order():
    with pytest.raises(ValueError, match="before"):
        splice(f"{END_MARKER}\n{START_MARKER}\n", "table")


PROVENANCE_RESULTS = {
    **RESULTS,
    "n_folds": 5,
    "n_rows": 307511,
    "n_features_used": 657,
    "selected_features_source": "params/selected_features.json",
    "selected_features_sha256": "abc123def4567890" + "0" * 48,
}


def test_render_provenance_names_the_run_behind_the_table():
    line = render_provenance(PROVENANCE_RESULTS)
    assert "5-fold CV" in line
    assert "307,511 rows" in line
    assert "657 features from `params/selected_features.json`" in line
    assert "abc123def456" in line
    assert line.startswith("<sub>") and line.endswith("</sub>")


def test_render_provenance_reports_an_unpinned_feature_set_as_such():
    line = render_provenance({**PROVENANCE_RESULTS, "selected_features_source": "all"})
    assert "657 features from all engineered features" in line
    assert "`all`" not in line


def test_render_provenance_is_empty_for_a_pre_provenance_results_file():
    assert render_provenance(RESULTS) == ""


HOLDOUT_RESULTS = {
    **PROVENANCE_RESULTS,
    "n_holdout": 61503,
    "holdout_frac": 0.2,
    "lgbm_holdout_auc": 0.78512,
    "xgb_holdout_auc": 0.78604,
    "catboost_holdout_auc": 0.78488,
    "ensemble_holdout_auc": 0.78901,
}


def test_render_table_adds_a_holdout_column_when_the_run_produced_one():
    table = render_table(HOLDOUT_RESULTS)

    assert "| Model | OOF AUC | Holdout AUC | Weight |" in table
    assert "| LightGBM | 0.78791 | 0.78512 | 0.14 |" in table
    assert "| **Ensemble** | **0.79348** | **0.78901** | — |" in table


def test_render_table_keeps_the_old_shape_without_a_holdout():
    table = render_table(RESULTS)

    assert "Holdout" not in table
    assert "| LightGBM | 0.78791 | 0.14 |" in table


def test_render_provenance_leads_with_the_holdout():
    line = render_provenance(HOLDOUT_RESULTS)

    assert line.startswith("<sub>61,503-row holdout (20%) split off before any CV")
    assert "5-fold CV" in line


def test_render_provenance_omits_a_holdout_that_was_not_taken():
    line = render_provenance({**HOLDOUT_RESULTS, "n_holdout": 0})

    assert "holdout" not in line
    assert "5-fold CV" in line


def test_check_mode_reports_a_stale_table_without_writing(tmp_path, monkeypatch):
    import json

    import update_readme

    readme = tmp_path / "README.md"
    readme.write_text(f"# Title\n\n{START_MARKER}\n\nstale\n\n{END_MARKER}\n",
                      encoding="utf-8")
    predictions = tmp_path / "predictions"
    predictions.mkdir()
    (predictions / "results.json").write_text(json.dumps(HOLDOUT_RESULTS))

    monkeypatch.setattr(update_readme, "PREDICTIONS_DIR", predictions)
    monkeypatch.setattr(update_readme, "ROOT_DIR", tmp_path)
    before = readme.read_text(encoding="utf-8")

    monkeypatch.setattr("sys.argv", ["update_readme.py", "--check"])
    assert update_readme.main() == 1
    assert readme.read_text(encoding="utf-8") == before

    monkeypatch.setattr("sys.argv", ["update_readme.py"])
    assert update_readme.main() == 0
    assert "0.78901" in readme.read_text(encoding="utf-8")

    monkeypatch.setattr("sys.argv", ["update_readme.py", "--check"])
    assert update_readme.main() == 0
