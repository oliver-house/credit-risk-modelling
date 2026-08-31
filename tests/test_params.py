import json

import pytest

from src.config import LGBM_PARAMS, PARAMS_DIR
from src.models.lgbm_model import _BEST_PARAMS_PATH, _load_params


def test_committed_params_file_loads_and_is_trainable():
    assert _BEST_PARAMS_PATH.exists(), "params/lgbm_best_params.json should be committed"
    params = _load_params()

    assert params["objective"] == "binary"
    assert params["metric"] == "auc"
    assert "n_estimators" in params

    assert params.get("subsample_freq", 0) > 0


def test_tuned_values_override_the_config_defaults():
    with open(_BEST_PARAMS_PATH) as f:
        payload = json.load(f)
    tuned = payload.get("params", payload)

    params = _load_params()
    for key, value in tuned.items():
        assert params[key] == value, f"tuned {key} was not applied over the default"


def test_both_on_disk_schemas_are_accepted(tmp_path, monkeypatch):
    flat = {"learning_rate": 0.123}
    nested = {"oof_auc": 0.78, "params": {"learning_rate": 0.123}}

    for payload in (flat, nested):
        path = tmp_path / "lgbm_best_params.json"
        path.write_text(json.dumps(payload))
        monkeypatch.setattr("src.models.lgbm_model._BEST_PARAMS_PATH", path)

        params = _load_params()
        assert params["learning_rate"] == 0.123
        assert params["objective"] == "binary"


def test_missing_file_falls_back_to_config_defaults(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "src.models.lgbm_model._BEST_PARAMS_PATH", tmp_path / "does_not_exist.json"
    )
    assert _load_params() == LGBM_PARAMS


def test_empty_params_file_is_rejected(tmp_path, monkeypatch):
    path = tmp_path / "lgbm_best_params.json"
    path.write_text("{}")
    monkeypatch.setattr("src.models.lgbm_model._BEST_PARAMS_PATH", path)

    with pytest.raises(ValueError, match="no usable params"):
        _load_params()


def test_selected_features_file_is_committed_and_non_empty():
    selected_path = PARAMS_DIR / "selected_features.json"
    assert selected_path.exists()
    with open(selected_path) as f:
        selected = json.load(f)
    assert isinstance(selected, list) and selected
    assert len(selected) == len(set(selected))
