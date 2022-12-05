import itertools
from typing import Any, Literal, TypeAlias

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import Pipeline

ExperimentState: TypeAlias = Literal["clear", "fit"]
CovarianceType: TypeAlias = Literal["full", "diag", "tied", "spherical"]


class GMMExperiment:
    """Encapsulates the steps to perform model selection with a
    Gaussian Mixture.
    """

    def __init__(
        self,
        data: pd.DataFrame | np.ndarray,
        *,
        preprocessor: Any,
        max_components: int,
        **kwargs,
    ) -> None:
        """Initialises the experiment."""

        self.data: pd.DataFrame | np.ndarray = data
        self.preprocessor = preprocessor

        self._components: np.ndarray = np.arange(2, max_components + 1)

        # kwargs
        self._random_state: int = kwargs.pop("random_state", None)

        # state
        self._covariances: list[CovarianceType] = ["spherical", "tied", "diag", "full"]

        self._state: ExperimentState = "clear"

        # other methods
        self._params: pd.DataFrame | None = None
        self._best_params: dict[str, int | str] | None = None
        self._colormap = ["navy", "turquoise", "cornflowerblue", "darkorange"]
        self._coloriter = itertools.cycle(self._colormap)
        self._figsize = kwargs.pop("figsize", (12, 12))

    def _bic_get(self, return_best_params: bool = True, **kwargs):
        """Compute the BIC of an individual combination of hyperparameters."""

        n_components = kwargs.pop("n_components", None)
        covariance_type = kwargs.pop("covariance_type", None)

        prep = self.preprocessor
        random_state = self._random_state

        interim_data = prep.fit_transform(self.data)

        model = GaussianMixture(
            n_components=n_components,
            covariance_type=covariance_type,
            random_state=random_state,
        )

        _ = model.fit(interim_data)

        bic = model.bic(interim_data)

        if return_best_params:
            return n_components, covariance_type, bic
        return bic

    def bics_get(self) -> pd.DataFrame:
        """Return an ordered DataFrame of BIC scores computed using the
        Experiment's n_components range."""

        results = [
            self._bic_get(
                data=self.data,
                return_best_params=True,
                n_components=comp,
                covariance_type=cov,
            )
            for comp in self._components
            for cov in self._covariances
        ]

        self._state = "fit"

        self._params = pd.DataFrame(
            data=results, columns=["n_components", "covariance_type", "bic"]
        ).sort_values("bic", ascending=True)

        return self._params

    def bics_plot_(self) -> None:
        """Just see here: https://scikit-learn.org/stable/auto_examples/mixture/plot_gmm_selection.html#sphx-glr-auto-examples-mixture-plot-gmm-selection-py

        Can/should/must be optimised.
        """
        if self._state == "clear":
            raise NotFittedError("call 'get_bics' before this")

        bars = []
        bic = self._params["bic"].values

        plt.figure(figsize=self._figsize)
        spl = plt.subplot(2, 1, 1)

        for i, (cv_type, color) in enumerate(zip(self._covariances, self._coloriter)):
            xpos = self._components + 0.2 * (i - 2)
            bars.append(
                plt.bar(
                    xpos,
                    bic[i * len(self._components) : (i + 1) * len(self._components)],
                    width=0.2,
                    color=color,
                )
            )

        plt.xticks(self._components)
        plt.ylim([bic.min() * 1.01 - 0.01 * bic.max(), bic.max()])
        plt.title("BIC score per model")

        xpos = (
            np.mod(bic.argmin(), len(self._components))
            + 0.65
            + 0.2 * np.floor(bic.argmin() / len(self._components))
        )

        plt.text(xpos, bic.min() * 0.97 + 0.03 * bic.max(), "*", fontsize=14)
        spl.set_xlabel("Number of components")
        spl.legend([b[0] for b in bars], self._covariances)

    def best_params_get_(self) -> dict[str, str | int]:
        """Get the best parameters to fit a new model."""
        if self._state == "clear":
            raise NotFittedError("call 'get_bics' before this")

        self._best_params = self._params.iloc[0, 0:2].to_dict()

        return self._best_params

    def best_model_get_(self) -> Pipeline:
        """Fit the best GMM chosen using the BIC criterion."""
        if self._state == "clear":
            raise NotFittedError("call 'get_bics' before this")

        if self._best_params is None:
            self.best_params_get_()

        best_model = GaussianMixture(
            n_components=self._best_params.get("n_components"),
            covariance_type=self._best_params.get("covariance_type"),
        )

        return Pipeline([("preprocessor", self.preprocessor), ("gmm", best_model)])
