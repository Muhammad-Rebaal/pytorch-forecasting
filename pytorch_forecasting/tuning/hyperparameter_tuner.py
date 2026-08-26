"""Hyperparameter tuning for v2 forecasters"""

import inspect

from lightning.pytorch.core.datamodule import LightningDataModule
import torch.nn as nn

from pytorch_forecasting.data import TimeSeries


class HyperparameterTuner:
    def __init__(
        self,
        model,
        data,
        trainer_cfg=None,
        datamodule_cfg=None,
        **fixed_hparams,
    ):
        """Set up the tuner, build the datamodule once, and validate inputs.

        Parameters
        ----------
        model : str or type
            Model name (e.g. ``"TFT"``) or the package class directly.
        data : TimeSeries or LightningDataModule
            Dataset to split and reuse across all trials.
        trainer_cfg : dict, optional
            Extra kwargs forwarded to ``lightning.Trainer``.
        datamodule_cfg : dict, optional
            Extra kwargs forwarded to the model's datamodule constructor.
        **fixed_hparams
            Any model parameter that should stay constant, e.g.
            ``hidden_size=128``.
        """
        self.pkg_cls = self._resolve_pkg_cls(model)
        self.fixed_hparams = fixed_hparams
        self.trainer_cfg = trainer_cfg or {}
        self.datamodule_cfg = datamodule_cfg or {}
        self._validate_fixed_hparams()

        if isinstance(data, TimeSeries):
            temp_pkg = self.pkg_cls(datamodule_cfg=self.datamodule_cfg)
            self.datamodule = temp_pkg._build_datamodule(data)
            self.datamodule.setup(stage="fit")
        elif isinstance(data, LightningDataModule):
            data.setup("fit")
            self.datamodule = data
        else:
            raise ValueError("data must be a TimeSeries or LightningDataModule")

    def _resolve_pkg_cls(self, model_name_or_cls):
        if isinstance(model_name_or_cls, str):
            from pytorch_forecasting._registry._lookup import all_objects

            # Iterate through the registry to find the class with the matching tag
            for name, cls in all_objects(
                return_names=True, object_types="forecaster_pytorch_v2"
            ):
                if cls.get_class_tags().get("info:name", "") == model_name_or_cls:
                    return cls
            raise ValueError(
                f"Could not find a v2 model with name: {model_name_or_cls}"
            )
        else:
            return model_name_or_cls

    def _validate_fixed_hparams(self):
        """Raise early if the user passed a parameter name
        that the model doesn't accept.
        """
        sig = inspect.signature(self.pkg_cls.get_cls().__init__)
        valid_param_names = sig.parameters.keys()

        for param_name in self.fixed_hparams:
            if param_name not in valid_param_names and param_name != "self":
                raise ValueError(
                    f"'{param_name}' is not a valid parameter of "
                    f"{self.pkg_cls.get_cls().__name__}. "
                    f"Valid parameters: {list(valid_param_names)}"
                )

    def _discover_hyperparameters(self, trial, custom_ranges=None):
        """Build a sampled config dict for one Optuna trial.

        Reads the model's ``__init__`` signature, skips base-class and
        user-fixed params, then picks values using ``custom_ranges`` when
        given or simple heuristics based on the default value's type.

        Parameters
        ----------
        trial : optuna.Trial
            Current trial whose ``suggest_*`` methods are called.
        custom_ranges : dict, optional
            Per-param overrides, e.g. ``{"dropout": (0.1, 0.5)}`` or
            ``{"hidden_size": [32, 64, 128]}``.

        Returns
        -------
        dict
            Sampled hyperparameters for this trial.
        """
        import typing

        from pytorch_forecasting.models.base._base_model_v2 import BaseModel

        base_params = set(inspect.signature(BaseModel.__init__).parameters.keys())
        sig = inspect.signature(self.pkg_cls.get_cls().__init__)

        sampled_cfg = {}
        custom_ranges = custom_ranges or {}

        for param_name, param_obj in sig.parameters.items():
            if param_name in base_params or param_name == "self":
                continue

            if param_name in self.fixed_hparams:
                continue

            if param_name in custom_ranges:
                r = custom_ranges[param_name]
                if isinstance(r, list):
                    sampled_cfg[param_name] = trial.suggest_categorical(param_name, r)
                else:
                    low, high = r
                    if (
                        isinstance(param_obj.default, int)
                        and type(param_obj.default) is not bool
                    ):
                        sampled_cfg[param_name] = trial.suggest_int(
                            param_name, low, high
                        )
                    else:
                        sampled_cfg[param_name] = trial.suggest_float(
                            param_name, low, high
                        )
                continue

            default = param_obj.default

            if (
                default is inspect.Parameter.empty
                or default is None
                or isinstance(default, (dict, list))
            ):
                continue

            if type(default) is bool:
                sampled_cfg[param_name] = trial.suggest_categorical(
                    param_name, [True, False]
                )

            elif isinstance(default, int):
                if default <= 8:
                    low = max(0, default - 2) if default >= 0 else default - 2
                    high = default + 2
                    sampled_cfg[param_name] = trial.suggest_int(param_name, low, high)
                else:
                    low = max(1, default // 4)
                    high = default * 4
                    sampled_cfg[param_name] = trial.suggest_int(
                        param_name, low, high, log=True
                    )

            elif isinstance(default, float):
                if default <= 1.0:
                    low = max(0.0, default / 2)
                    high = min(1.0, default * 3) if default > 0.0 else 0.1
                    sampled_cfg[param_name] = trial.suggest_float(param_name, low, high)
                else:
                    low = default / 4
                    high = default * 4
                    sampled_cfg[param_name] = trial.suggest_float(
                        param_name, low, high, log=True
                    )

            elif typing.get_origin(param_obj.annotation) is typing.Literal:
                choices = list(typing.get_args(param_obj.annotation))
                sampled_cfg[param_name] = trial.suggest_categorical(param_name, choices)

        return sampled_cfg

    def optimize(
        self,
        n_trials=100,
        timeout=3600 * 8,
        max_epochs=20,
        custom_ranges=None,
        study=None,
        direction="minimize",
    ):
        """Run the Optuna study.

        Parameters
        ----------
        n_trials : int
            How many trials to run.
        timeout : float
            Wall-clock budget in seconds (default 8 h).
        max_epochs : int
            Training epochs per trial.
        custom_ranges : dict, optional
            Per-param search ranges forwarded to
            :meth:`_discover_hyperparameters`.
        study : optuna.Study, optional
            Existing study to resume; a new one is created if ``None``.
        direction : str
            ``"minimize"`` or ``"maximize"``.

        Returns
        -------
        optuna.Study
            Completed study with all trial results.
        """
        from skbase.utils.dependencies import _safe_import

        optuna = _safe_import("optuna")

        def _objective(trial):
            sampled_cfg = self._discover_hyperparameters(trial, custom_ranges)
            model_cfg = {**sampled_cfg, **self.fixed_hparams}
            if "loss" not in model_cfg:
                model_cfg["loss"] = nn.MSELoss()
            trainer_cfg = {
                **self.trainer_cfg,
                "max_epochs": max_epochs,
                "enable_progress_bar": False,
            }
            pkg = self.pkg_cls(
                model_cfg=model_cfg,
                trainer_cfg=trainer_cfg,
                datamodule_cfg=self.datamodule_cfg,
            )
            pkg.fit(self.datamodule, save_ckpt=False)

            metrics = pkg.trainer.callback_metrics
            for key in ("val_loss", "train_loss_epoch", "train_loss"):
                if key in metrics:
                    return metrics[key].item()
            raise RuntimeError(
                "No loss metric found in trainer.callback_metrics. "
                f"Available keys: {list(metrics.keys())}"
            )

        if study is None:
            study = optuna.create_study(direction=direction)

        study.optimize(_objective, n_trials=n_trials, timeout=timeout)

        return study
