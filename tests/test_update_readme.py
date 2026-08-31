import pytest

from update_readme import END_MARKER, START_MARKER, render_table, splice

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
