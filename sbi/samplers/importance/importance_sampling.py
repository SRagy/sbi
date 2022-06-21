from math import sqrt
from typing import Callable, Tuple, Optional
import torch
from torch import Tensor, Size
from torch.distributions import Distribution


def importance_sample(
    potential_fn,
    proposal,
    num_samples: int = 1,
    importance_weight_smoothing: str = "raw",
) -> Tuple[Tensor, Tensor, Tensor]:
    """Returns samples from proposal, log(importance weights), and log(Z).

    Args:
        potential_fn: Unnormalized potential function.
        proposal: Proposal distribution with `.sample()` and `.log_prob()` methods.
        num_samples: Number of samples to draw.

    Returns:
        Samples and importance weights.
    """
    samples = proposal.sample((num_samples,))

    potential_logprobs = potential_fn.log_prob(samples)
    proposal_logprobs = proposal.log_prob(samples)
    log_importance_weights = potential_logprobs - proposal_logprobs

    if importance_weight_smoothing == "psis":
        largest_weights = select_largest_weights(log_weights)
        k, sigma = psis(log_weights=largest_weights)
        pareto_dist = pass
        corrected_weights = 

    log_norm_constant = torch.mean(log_importance_weights.exp()).log()
    return samples, log_importance_weights, log_norm_constant

def select_largest_weights(log_weights: Tensor) -> Tensor:
    M = int(min(len(log_weights) / 5, 3 * sqrt(len(log_weights))))
    with torch.no_grad():
        log_weights = log_weights[torch.isfinite(log_weights)]
        logweights_max = log_weights.max()
        weights = torch.exp(log_weights - logweights_max)  # Thus will only affect scale
        vals, _ = weights.sort()
        largest_weigths = vals[-M:]
    return largest_weigths

def psis(log_weights: Tensor) -> Tuple[float, float]:
    r"""This will evaluate the posteriors quality by investingating its importance
    weights. If q is a perfect posterior approximation then $q(\theta) \propto
    p(\theta, x_o)$ thus $\log w(\theta) = \log \frac{p(\theta, x_o)}{\log q(\theta)} =
    \log p(x_o)$ is constant. This function will fit a Generalized Paretto
    distribution to the tails of w. The shape parameter k serves as metric as detailed
    in [1]. In short it is related to the variance of a importance sampling estimate,
    especially for k > 1 the variance will be infinite.

    NOTE: In our experience this metric does distinguish "very bad" from "ok", but
    becomes less sensitive to distinguish "ok" from "good".

    Args:
        potential_function: Potential function of target.
        q: Variational distribution, should be proportional to the potential_function
        proposal: Proposal for samples. Typically this is q.
        N: Number of samples involved in the test.

    Returns:
        float: Quality metric

    Reference:
        [1] _Yes, but Did It Work?: Evaluating Variational Inference_, Yuling Yao, Aki
        Vehtari, Daniel Simpson, Andrew Gelman, 2018, https://arxiv.org/abs/1802.02538

    """
    return gpdfit(weights)


def gpdfit(
    x: Tensor, sorted: bool = True, eps: float = 1e-8, return_quadrature: bool = False
) -> Tuple:
    """Maximum aposteriori estimate of a Generalized Paretto distribution.

    Pytorch version of gpdfit according to
    https://github.com/avehtari/PSIS/blob/master/py/psis.py. This function will compute
    a MAP (more stable than the MLE estimator).


    Args:
        x: Tensor of floats, the data which is used to fit the GPD.
        sorted: If x is already sorted
        eps: Numerical stability jitter
        return_quadrature: Weather to return individual results.
    Returns:
        Tuple: Parameters of the Generalized Paretto Distribution.

    """
    if not sorted:
        x, _ = x.sort()
    N = len(x)
    PRIOR = 3
    M = 30 + int(sqrt(N))

    bs = torch.arange(1, M + 1, device=x.device)
    bs = 1 - torch.sqrt(M / (bs - 0.5))
    bs /= PRIOR * x[int(N / 4 + 0.5) - 1]
    bs += 1 / x[-1]

    ks = -bs
    temp = ks[:, None] * x
    ks = torch.log1p(temp).mean(dim=1)
    L = N * (torch.log(-bs / ks) - ks - 1)

    temp = torch.exp(L - L[:, None])
    w = 1 / torch.sum(temp, dim=1)

    dii = w >= 10 * eps
    if not torch.all(dii):
        w = w[dii]
        bs = bs[dii]
    w /= w.sum()

    # posterior mean for b
    b = torch.sum(bs * w)
    # Estimate for k
    temp = (-b) * x
    temp = torch.log1p(temp)
    k = torch.mean(temp)
    if return_quadrature:
        temp = -x
        temp = bs[:, None] * temp
        temp = torch.log1p(temp)
        ks = torch.mean(temp, dim=1)

    # estimate for sigma
    sigma = -k / b * N / (N - 0)
    # weakly informative prior for k
    a = 10
    k = k * N / (N + a) + a * 0.5 / (N + a)
    if return_quadrature:
        ks *= N / (N + a)
        ks += a * 0.5 / (N + a)

    if return_quadrature:
        return k, sigma, ks, w
    else:
        return k, sigma
