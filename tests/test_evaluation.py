import numpy as np
import pytest

from src.evaluation import (
    brier_score,
    calibration_slope,
    calibration_table,
    expected_cost,
    fit_logistic_baseline,
    gini,
    ks_statistic,
    ks_threshold,
    optimal_threshold,
    score_bands,
    summarise,
    threshold_sensitivity,
)


@pytest.fixture
def perfect():
    y = np.array([0] * 50 + [1] * 50)
    return y, y * 0.9 + 0.05


@pytest.fixture
def population():
    rng = np.random.default_rng(3)
    n = 4_000
    latent = rng.normal(size=n)
    y = (rng.random(n) < 1 / (1 + np.exp(-(1.5 * latent - 2.7)))).astype(int)
    scores = 1 / (1 + np.exp(-(1.2 * latent - 2.7)))
    return y, scores



def test_gini_is_twice_auc_minus_one(perfect):
    y, scores = perfect
    assert gini(y, scores) == pytest.approx(1.0)
    assert gini(y, 1 - scores) == pytest.approx(-1.0)


def test_ks_is_one_for_a_perfect_separator(perfect):
    y, scores = perfect
    assert ks_statistic(y, scores) == pytest.approx(1.0)
    assert ks_threshold(y, scores) == pytest.approx(0.95)


def test_ks_is_near_zero_for_a_worthless_score():
    rng = np.random.default_rng(0)
    y = np.tile([0, 1], 2_000)
    assert ks_statistic(y, rng.random(len(y))) < 0.1


def test_summarise_reports_the_headline_numbers(population):
    y, scores = population
    report = summarise(y, scores, "ensemble")

    assert report["label"] == "ensemble"
    assert report["n"] == len(y)
    assert report["base_rate"] == pytest.approx(y.mean())
    assert report["gini"] == pytest.approx(2 * report["auc"] - 1)
    assert 0 < report["ks"] < 1
    assert 0 < report["brier"] < 0.25



def test_bands_partition_the_population(population):
    y, scores = population
    bands = score_bands(y, scores, n_bands=10)

    assert len(bands) == 10
    assert bands["n"].sum() == len(y)
    assert bands["n_bad"].sum() == y.sum()
    assert bands["cum_bad_capture"].iloc[-1] == pytest.approx(1.0)
    assert bands["cum_population"].iloc[-1] == pytest.approx(1.0)


def test_the_riskiest_band_comes_first_and_lifts(population):
    y, scores = population
    bands = score_bands(y, scores, n_bands=10)

    assert bands["band"].iloc[0] == 1
    assert bands["bad_rate"].iloc[0] > bands["bad_rate"].iloc[-1]
    assert bands["lift"].iloc[0] > 1 > bands["lift"].iloc[-1]
    assert bands["mean_score"].is_monotonic_decreasing


def test_a_perfect_score_puts_every_default_in_the_top_bands(perfect):
    y, scores = perfect
    bands = score_bands(y, scores, n_bands=2)

    assert bands["bad_rate"].tolist() == [1.0, 0.0]
    assert bands["cum_bad_capture"].iloc[0] == pytest.approx(1.0)


def test_bands_reject_mismatched_inputs():
    with pytest.raises(ValueError, match="labels against"):
        score_bands(np.array([0, 1]), np.array([0.1, 0.2, 0.3]))

    with pytest.raises(ValueError, match="at least 2"):
        score_bands(np.array([0, 1]), np.array([0.1, 0.2]), n_bands=1)



def test_calibration_of_a_perfectly_calibrated_score():
    rng = np.random.default_rng(11)
    p = rng.uniform(0.01, 0.99, 20_000)
    y = (rng.random(len(p)) < p).astype(int)

    table = calibration_table(y, p, n_bins=10)

    assert len(table) == 10
    assert table["n"].sum() == len(y)
    np.testing.assert_allclose(table["observed_rate"], table["mean_predicted"],
                               atol=0.03)
    assert calibration_slope(table) == pytest.approx(1.0, abs=0.1)


def test_a_systematically_understated_score_shows_a_positive_gap():
    rng = np.random.default_rng(12)
    p = rng.uniform(0.01, 0.5, 20_000)
    y = (rng.random(len(p)) < 2 * p).astype(int)

    table = calibration_table(y, p, n_bins=10)

    assert (table["gap"] > 0).all()
    assert calibration_slope(table) == pytest.approx(2.0, abs=0.2)


def test_brier_rewards_the_better_probabilities():
    y = np.array([0, 0, 1, 1])
    assert brier_score(y, np.array([0.0, 0.0, 1.0, 1.0])) == pytest.approx(0.0)
    assert brier_score(y, np.array([0.5, 0.5, 0.5, 0.5])) == pytest.approx(0.25)



def test_expected_cost_counts_both_error_types():
    y = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.9, 0.2, 0.8])

    assert expected_cost(y, scores, 0.5, cost_fn=10, cost_fp=1) == pytest.approx(11)
    assert expected_cost(y, scores, 1.1, cost_fn=10, cost_fp=1) == pytest.approx(20)
    assert expected_cost(y, scores, 0.0, cost_fn=10, cost_fp=1) == pytest.approx(2)


def test_the_optimal_threshold_finds_a_planted_clean_split():
    y = np.array([0] * 50 + [1] * 50)
    scores = np.concatenate([np.linspace(0.0, 0.4, 50), np.linspace(0.6, 1.0, 50)])

    decision = optimal_threshold(y, scores, cost_fn=10, cost_fp=1)

    assert decision.expected_cost == pytest.approx(0.0)
    assert 0.4 < decision.threshold <= 0.6
    assert decision.approval_rate == pytest.approx(0.5)
    assert decision.bad_rate_approved == pytest.approx(0.0)
    assert decision.bad_capture == pytest.approx(1.0)
    assert decision.n_approved + decision.n_declined == len(y)


def test_the_optimal_threshold_beats_every_other_cut(population):
    y, scores = population
    decision = optimal_threshold(y, scores, cost_fn=10, cost_fp=1)

    for candidate in np.quantile(scores, np.linspace(0, 1, 25)):
        assert expected_cost(y, scores, candidate, 10, 1) >= decision.expected_cost - 1e-9


def test_a_harsher_loss_assumption_declines_more(population):
    y, scores = population

    lenient = optimal_threshold(y, scores, cost_fn=2, cost_fp=1)
    harsh = optimal_threshold(y, scores, cost_fn=50, cost_fp=1)

    assert harsh.threshold < lenient.threshold
    assert harsh.approval_rate < lenient.approval_rate
    assert harsh.bad_capture > lenient.bad_capture


def test_threshold_sensitivity_spans_the_ratios(population):
    y, scores = population
    table = threshold_sensitivity(y, scores, ratios=(2, 10, 50))

    assert table["cost_ratio"].tolist() == ["2:1", "10:1", "50:1"]
    assert table["approval_rate"].is_monotonic_decreasing
    assert table["bad_capture"].is_monotonic_increasing



def test_the_logistic_baseline_cross_fits_and_scores_a_holdout():
    rng = np.random.default_rng(5)
    X = rng.normal(size=(600, 6))
    y = (rng.random(600) < 1 / (1 + np.exp(-(2 * X[:, 0] - X[:, 1])))).astype(int)
    X_holdout = rng.normal(size=(150, 6))

    oof, holdout = fit_logistic_baseline(X, y, X_holdout, n_folds=3)

    assert oof.shape == (600,)
    assert holdout.shape == (150,)
    assert ((oof > 0) & (oof < 1)).all()
    assert gini(y, oof) > 0.5


def test_the_logistic_baseline_tolerates_missing_values():
    rng = np.random.default_rng(6)
    X = rng.normal(size=(400, 4))
    y = (X[:, 0] + rng.normal(scale=0.5, size=400) > 0).astype(int)
    X[rng.random(X.shape) < 0.2] = np.nan

    oof, _ = fit_logistic_baseline(X, y, n_folds=3)

    assert np.isfinite(oof).all()


def test_the_baseline_returns_nothing_for_a_holdout_it_was_not_given():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(200, 3))
    y = (X[:, 0] > 0).astype(int)

    oof, holdout = fit_logistic_baseline(X, y, n_folds=2)

    assert holdout is None
    assert oof.shape == (200,)


def test_render_bands_produces_one_markdown_row_per_band(population):
    from evaluate import render_bands

    y, scores = population
    table = render_bands(score_bands(y, scores, n_bands=5))

    assert table.count("\n") == 6
    assert "| Band |" in table
    assert table.splitlines()[2].startswith("| 1 |")
