import builtins
import json

import pytest

from src.tracking import RunTracker, flatten, tracking_uri

mlflow = pytest.importorskip("mlflow", reason="mlflow-skinny is an optional extra")


@pytest.fixture
def store(tmp_path):
    return {"db_path": tmp_path / "mlflow.db", "artifact_dir": tmp_path / "artifacts"}


@pytest.fixture(autouse=True)
def isolate_active_run():
    yield
    if mlflow.active_run() is not None:
        mlflow.end_run()


def _client(store):
    from mlflow.tracking import MlflowClient
    return MlflowClient(tracking_uri=tracking_uri(store["db_path"]))


def test_flatten_builds_dotted_param_names():
    assert flatten("lgbm", {"learning_rate": 0.05, "num_leaves": 34}) == {
        "lgbm.learning_rate": 0.05,
        "lgbm.num_leaves": 34,
    }
    assert flatten("", {"a": {"b": 1}}) == {"a.b": 1}
    assert flatten("n_folds", 5) == {"n_folds": 5}


def test_tracking_uri_is_a_sqlite_url(tmp_path):
    uri = tracking_uri(tmp_path / "mlflow.db")

    assert uri.startswith("sqlite:///")
    assert uri.endswith("mlflow.db")


def test_a_run_records_params_metrics_and_artefacts(store, tmp_path):
    artefact = tmp_path / "results.json"
    artefact.write_text(json.dumps({"ensemble_oof_auc": 0.79}))

    with RunTracker(run_name="unit-test", experiment="unit-tests", **store) as tracker:
        assert tracker.enabled
        tracker.log_params({"n_folds": 5, "selected_features_source": "params/x.json"})
        tracker.log_params({"learning_rate": 0.05}, prefix="lgbm")
        tracker.log_metrics({"ensemble_oof_auc": 0.79373, "ensemble_holdout_auc": 0.789})
        tracker.log_artifact(artefact)
        run_id = tracker.run_id

    run = _client(store).get_run(run_id)
    assert run.data.params["n_folds"] == "5"
    assert run.data.params["lgbm.learning_rate"] == "0.05"
    assert run.data.metrics["ensemble_oof_auc"] == pytest.approx(0.79373)
    assert run.info.status == "FINISHED"
    assert [a.path for a in _client(store).list_artifacts(run_id)] == ["results.json"]


def test_successive_runs_accumulate_rather_than_overwrite(store):
    for auc in (0.790, 0.793, 0.795):
        with RunTracker(experiment="history", **store) as tracker:
            tracker.log_metrics({"ensemble_oof_auc": auc})

    client = _client(store)
    experiment = client.get_experiment_by_name("history")
    runs = client.search_runs([experiment.experiment_id])

    assert len(runs) == 3
    assert sorted(r.data.metrics["ensemble_oof_auc"] for r in runs) == [0.790, 0.793, 0.795]


def test_non_numeric_values_are_kept_out_of_the_metrics(store):
    with RunTracker(experiment="types", **store) as tracker:
        tracker.log_metrics({"auc": 0.79, "source": "params/x.json", "smoke": True})
        run_id = tracker.run_id

    metrics = _client(store).get_run(run_id).data.metrics
    assert set(metrics) == {"auc"}


def test_a_failing_run_is_recorded_as_failed(store):
    with (
        pytest.raises(RuntimeError, match="training blew up"),
        RunTracker(experiment="failures", **store) as tracker,
    ):
        tracker.log_metrics({"auc": 0.5})
        run_id = tracker.run_id
        raise RuntimeError("training blew up")

    assert _client(store).get_run(run_id).info.status == "FAILED"


def test_disabling_tracking_is_a_silent_no_op(store, tmp_path):
    with RunTracker(enabled=False, experiment="disabled", **store) as tracker:
        assert not tracker.enabled
        assert tracker.run_id is None
        tracker.log_params({"n_folds": 5})
        tracker.log_metrics({"auc": 0.79})
        tracker.log_artifact(tmp_path / "does_not_exist.json")

    assert not store["db_path"].exists()


def test_a_missing_mlflow_degrades_to_a_no_op(monkeypatch, store):
    real_import = builtins.__import__

    def no_mlflow(name, *args, **kwargs):
        if name == "mlflow":
            raise ImportError("No module named 'mlflow'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_mlflow)

    with RunTracker(experiment="absent", **store) as tracker:
        assert not tracker.enabled
        tracker.log_metrics({"auc": 0.79})


def test_an_mlflow_error_mid_run_disables_tracking_without_raising(store):
    with RunTracker(experiment="errors", **store) as tracker:
        assert tracker.enabled
        tracker._mlflow = None
        tracker.log_metrics({"auc": 0.79})
        assert not tracker.enabled


def test_a_missing_artefact_is_warned_about_not_raised(store, tmp_path):
    with RunTracker(experiment="artefacts", **store) as tracker:
        tracker.log_artifact(tmp_path / "never_written.png")
        run_id = tracker.run_id
        assert tracker.enabled

    assert _client(store).list_artifacts(run_id) == []
