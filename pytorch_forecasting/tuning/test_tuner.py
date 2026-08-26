"""Tests for the v2 HyperparameterTuner."""

import numpy as np
import pandas as pd
import pytest
from skbase.utils.dependencies import _safe_import

optuna = _safe_import("optuna")

from pytorch_forecasting.data import TimeSeries
from pytorch_forecasting.tuning.hyperparameter_tuner import HyperparameterTuner


@pytest.fixture
def dummy_ts():
    """Minimal TimeSeries object for initializing the tuner."""
    n_samples = 200
    n_groups = 3
    dfs = []
    for g in range(n_groups):
        time_idx = np.arange(n_samples)
        values = np.sin(2 * np.pi * time_idx / 20) + np.random.normal(0, 0.1, n_samples)
        dfs.append(
            pd.DataFrame(
                {
                    "time_idx": time_idx,
                    "series_id": g,
                    "value": values,
                }
            )
        )

    df = pd.concat(dfs, ignore_index=True)

    return TimeSeries(
        df,
        time="time_idx",
        group=["series_id"],
        target=["value"],
        num=[],
        cat=[],
        known=["time_idx"],
        unknown=["value"],
    )


@pytest.fixture
def dm_cfg():
    """DataModule kwargs that work with TFT's EncoderDecoderTimeSeriesDataModule."""
    return {"max_encoder_length": 10, "max_prediction_length": 5, "batch_size": 4}


@pytest.fixture
def tuner(dummy_ts, dm_cfg):
    """Plain tuner with no fixed hparams."""
    return HyperparameterTuner(model="TFT", data=dummy_ts, datamodule_cfg=dm_cfg)


@pytest.fixture
def tuner_with_fixed(dummy_ts, dm_cfg):
    """Tuner with hidden_size pinned to 128."""
    return HyperparameterTuner(
        model="TFT", data=dummy_ts, datamodule_cfg=dm_cfg, hidden_size=128
    )


@pytest.fixture
def tuner_for_optimize(dummy_ts, dm_cfg):
    """Tuner with hidden_size and attention_head_size pinned to compatible values.

    MultiheadAttention requires hidden_size % attention_head_size == 0,
    so we fix them to avoid random sampling failures.
    """
    return HyperparameterTuner(
        model="TFT",
        data=dummy_ts,
        datamodule_cfg=dm_cfg,
        hidden_size=64,
        attention_head_size=4,
        output_size=1,
    )


@pytest.fixture
def trial():
    """Blank Optuna trial for calling _discover_hyperparameters."""
    study = optuna.create_study()
    return study.ask()


class TestResolvePkgCls:
    """Registry lookup from model name to package class."""

    def test_resolves_string_name(self, tuner):
        """'TFT' string should map to TFT_pkg_v2."""
        assert tuner.pkg_cls.__name__ == "TFT_pkg_v2"

    def test_resolves_class_directly(self, dummy_ts, dm_cfg):
        """Passing the class itself should skip the registry entirely."""
        from pytorch_forecasting.models.temporal_fusion_transformer._tft_pkg_v2 import (
            TFT_pkg_v2,
        )

        tuner = HyperparameterTuner(
            model=TFT_pkg_v2, data=dummy_ts, datamodule_cfg=dm_cfg
        )
        assert tuner.pkg_cls is TFT_pkg_v2

    def test_unknown_model_name_raises(self, dummy_ts, dm_cfg):
        """Vague name should raise ValueError."""
        with pytest.raises(ValueError, match="Could not find"):
            HyperparameterTuner(
                model="NonExistentModel", data=dummy_ts, datamodule_cfg=dm_cfg
            )


class TestValidateFixedHparams:
    """Fail-fast check for typos in fixed_hparams."""

    def test_valid_hparam_accepted(self, dummy_ts, dm_cfg):
        """Real param name should be stored without error."""
        tuner = HyperparameterTuner(
            model="TFT", data=dummy_ts, datamodule_cfg=dm_cfg, hidden_size=128
        )
        assert tuner.fixed_hparams["hidden_size"] == 128

    def test_invalid_hparam_raises(self, dummy_ts, dm_cfg):
        """Typo 'hiddne_size' should blow up immediately."""
        with pytest.raises(ValueError, match="hiddne_size"):
            HyperparameterTuner(
                model="TFT", data=dummy_ts, datamodule_cfg=dm_cfg, hiddne_size=128
            )

    def test_multiple_valid_hparams(self, dummy_ts, dm_cfg):
        """Several valid params at once should all land in fixed_hparams."""
        tuner = HyperparameterTuner(
            model="TFT",
            data=dummy_ts,
            datamodule_cfg=dm_cfg,
            hidden_size=128,
            dropout=0.2,
            num_layers=3,
        )
        assert tuner.fixed_hparams == {
            "hidden_size": 128,
            "dropout": 0.2,
            "num_layers": 3,
        }


class TestInitDataHandling:
    """Data path through __init__ (split once, tune many)."""

    def test_timeseries_creates_datamodule(self, tuner):
        """TimeSeries input should produce a stored datamodule."""
        assert tuner.datamodule is not None

    def test_invalid_data_type_raises(self, dm_cfg):
        """A plain list is not valid data."""
        with pytest.raises(ValueError, match="data must be"):
            HyperparameterTuner(model="TFT", data=[1, 2, 3], datamodule_cfg=dm_cfg)

    def test_default_cfgs_are_empty_dicts(self, dummy_ts, dm_cfg):
        """Omitting trainer_cfg should give an empty dict, not None."""
        tuner = HyperparameterTuner(model="TFT", data=dummy_ts, datamodule_cfg=dm_cfg)
        assert tuner.trainer_cfg == {}


class TestDiscoverHyperparameters:
    """Checks for the inspect-based search-space engine."""

    def test_fixed_params_excluded(self, tuner_with_fixed, trial):
        """Pinned params must not show up in the sampled config."""
        cfg = tuner_with_fixed._discover_hyperparameters(trial)
        assert "hidden_size" not in cfg

    def test_base_model_params_excluded(self, tuner, trial):
        """Infrastructure stuff (loss, optimizer, ...) should be skipped."""
        cfg = tuner._discover_hyperparameters(trial)
        for infra_param in ["loss", "optimizer", "optimizer_params", "lr_scheduler"]:
            assert infra_param not in cfg

    def test_float_param_discovered(self, tuner, trial):
        """dropout (float default 0.1) should come back as a float."""
        cfg = tuner._discover_hyperparameters(trial)
        assert "dropout" in cfg
        assert isinstance(cfg["dropout"], float)

    def test_int_param_discovered(self, tuner, trial):
        """hidden_size (int default 64) should come back as an int."""
        cfg = tuner._discover_hyperparameters(trial)
        assert "hidden_size" in cfg
        assert isinstance(cfg["hidden_size"], int)

    def test_custom_range_list_is_categorical(self, tuner, trial):
        """A list custom range should become a categorical choice."""
        cfg = tuner._discover_hyperparameters(
            trial, custom_ranges={"hidden_size": [32, 64, 128]}
        )
        assert cfg["hidden_size"] in [32, 64, 128]

    def test_custom_range_tuple_overrides_heuristic(self, tuner, trial):
        """A (low, high) tuple should override the default heuristic."""
        cfg = tuner._discover_hyperparameters(
            trial, custom_ranges={"dropout": (0.0, 1.0)}
        )
        assert 0.0 <= cfg["dropout"] <= 1.0

    def test_returns_dict(self, tuner, trial):
        """Return type is always a plain dict."""
        cfg = tuner._discover_hyperparameters(trial)
        assert isinstance(cfg, dict)

    def test_none_default_params_skipped(self, tuner, trial):
        """Params whose default is None (like metadata) get skipped."""
        cfg = tuner._discover_hyperparameters(trial)
        assert "metadata" not in cfg


class TestOptimize:
    """Test the optimize loop."""

    @pytest.mark.slow
    def test_optimize_returns_study(self, tuner_for_optimize):
        """Should hand back an optuna.Study with exactly one finished trial."""
        study = tuner_for_optimize.optimize(n_trials=1, max_epochs=1, timeout=120)
        assert isinstance(study, optuna.Study)
        assert len(study.trials) == 1
