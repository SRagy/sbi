from __future__ import annotations

from typing import Callable, Optional, Tuple

import numpy as np
import torch
from torch import Size
from torch.distributions import Distribution

from sbi.samplers.vi.vi_utils import gpdfit

_QUALITY_METRIC = {}
_METRIC_MESSAGE = {}


def register_quality_metric(
    cls: Optional[Callable] = None,
    name: Optional[str] = None,
    msg: Optional[str] = None,
) -> Callable:
    """This method will register a given metric for cheap quality evaluation of
    variational posteriors.

    Args:
        cls: Function to compute the metric.
        name: Associated name.
        msg: Short description on how to interpret the metric.

    """

    def _register(cls: Callable) -> Callable:
        if name is None:
            cls_name = cls.__name__
        else:
            cls_name = name
        if cls_name in _QUALITY_METRIC:
            raise ValueError(f"The metric {cls_name} is already registered")
        else:
            _QUALITY_METRIC[cls_name] = cls
            _METRIC_MESSAGE[cls_name] = msg
        return cls

    if cls is None:
        return _register
    else:
        return _register(cls)


def get_quality_metric(name: str) -> Tuple[Callable, str]:
    """Returns the quality, metric as well as a short description."""
    return _QUALITY_METRIC[name], _METRIC_MESSAGE[name]


def basic_checks(posterior, N: int = int(5e4)):
    """Makes some basic checks to ensure the distribution is well defined.

    Args:
        posterior: Variational posterior object to check. Of type `VIPosterior`. No
            typing due to circular imports.
        N: Number of samples that are checked.
    """
    prior = posterior._prior
    assert prior is not None, "Posterior has no `._prior` attribute."
    prior_samples = prior.sample(Size((N,)))
    samples = posterior.sample(Size((N,)))
    assert (torch.isfinite(samples)).all(), "Some of the samples are not finite"
    try:
        _ = prior.support
        has_support = True
    except (NotImplementedError, AttributeError):
        has_support = False
    if has_support:
        assert (
            prior.support.check(samples)  # type: ignore
        ).all(), "Some of the samples are not within the prior support."
    assert (
        torch.isfinite(posterior.log_prob(samples))
    ).all(), "The log probability is not finite for some samples"
    assert (
        torch.isfinite(posterior.log_prob(prior_samples))
    ).all(), "The log probability is not finite for some samples"


def proportional_to_joint_diagnostics(
    potential_function: Callable,
    q: Distribution,
    proposal: Optional[Distribution] = None,
    N: int = int(5e4),
) -> float:
    r"""This will evaluate the posteriors quality by investingating its importance
    weights. If q is a perfect posterior approximation then $q(\theta) \propto
    p(\theta, x_o)$. Thus we should be able to fit a line to $(q(\theta),
    p(\theta, x_o))$, whereas the slope will be proportional to the normalizing
    constant. The quality of a linear fit is hence a direct metric for the quality of q.
    We use R2 statistic.

    NOTE: In our experience this metric does distinguish "good" from "ok", but
    becomes less sensitive to distinguish "very bad" from "ok".

    Args:
        potential_function: Potential function of target.
        q: Variational distribution, should be proportional to the potential_function
        proposal: Proposal for samples. Typically this is q.
        N: Number of samples involved in the test.

    Returns:
        float: Quality metric

    """

    with torch.no_grad():
        if proposal is None:
            samples = q.sample(Size((N,)))
        else:
            samples = proposal.sample(Size((N,)))
        log_q = q.log_prob(samples)
        log_potential = potential_function(samples)

        X = log_q.exp().unsqueeze(-1)
        Y = log_potential.exp().unsqueeze(-1)
        w = torch.linalg.solve(X.T @ X, X.T @ Y)  # Linear regression

        residuals = Y - w * X
        var_res = torch.sum(residuals**2)
        var_tot = torch.sum((Y - Y.mean()) ** 2)
        r2 = 1 - var_res / var_tot  # R2 statistic to evaluate fit
    return r2.item()


@register_quality_metric(
    name="prop",
    msg="\t Good: Larger than 0.5, best is 1.0  Bad: Smaller than 0.5 \t \
        NOTE: Less sensitive to mode collapse.",
)
def proportionality(posterior, N: int = int(5e4)):
    """
    Args:
        posterior: Of type `VIPosterior`. No typing due to circular imports.
    """
    basic_checks(posterior)
    return proportional_to_joint_diagnostics(posterior.potential_fn, posterior.q, N=N)


@register_quality_metric(
    name="prop_prior",
    msg="\t Good: Larger than 0.5, best is 1.0  Bad: Smaller than 0.5 \t \
        NOTE: Less sensitive to mode collapse.",
)
def proportionality_prior(posterior, N: int = int(5e4)):
    """
    Args:
        posterior: Of type `VIPosterior`. No typing due to circular imports.
    """
    basic_checks(posterior)
    return proportional_to_joint_diagnostics(
        posterior.potential_fn, posterior.q, proposal=posterior._prior, N=N
    )


assert proportional_to_joint_diagnostics.__doc__ is not None
proportionality.__doc__ = proportional_to_joint_diagnostics.__doc__.split("Args:")[0]
proportionality_prior.__doc__ = proportional_to_joint_diagnostics.__doc__.split(
    "Args:"
)[0]
