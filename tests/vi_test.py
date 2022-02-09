# TODO ESPECIALLY THE VI METHODS AND VARIATIONAL FAMILIES WHEN FINSIHED SHOULD BE
# THESTED HERE!!!!

from __future__ import annotations

import numpy as np
from pkg_resources import Distribution
import pytest
import torch
from torch import eye, ones, zeros
from copy import deepcopy

from sbi import utils
from sbi.inference import SNLE, SNPE, SNRE, prepare_for_sbi, simulate_for_sbi
from sbi.inference.potentials.base_potential import BasePotential
import torch.distributions.transforms as torch_tf
from torch.distributions import MultivariateNormal
from sbi.inference.posteriors import VIPosterior
from sbi.simulators.linear_gaussian import true_posterior_linear_gaussian_mvn_prior
from tests.test_utils import check_c2st


@pytest.mark.slow
@pytest.mark.parametrize("num_dim", (1, 2))
@pytest.mark.parametrize("vi_method", ("rKL", "fKL", "IW", "alpha"))
@pytest.mark.parametrize("sampling_method", ("naive", "sir"))
def test_c2st_vi_on_Gaussian(
    num_dim: int, vi_method: str, sampling_method: str, set_seed
):
    """Test VI on Gaussian, comparing to ground truth target via c2st.

    Args:
        num_dim: parameter dimension of the gaussian model
        vi_method: different vi methods
        sampling_method: Different sampling methods
        set_seed: fixture for manual seeding
    """
    num_samples = 500

    likelihood_shift = -1.0 * ones(num_dim)
    likelihood_cov = 0.3 * eye(num_dim)
    prior_mean = zeros(num_dim)
    prior_cov = eye(num_dim)

    x_o = zeros((1, num_dim))
    target_distribution = true_posterior_linear_gaussian_mvn_prior(
        x_o[0], likelihood_shift, likelihood_cov, prior_mean, prior_cov
    )
    target_samples = target_distribution.sample((num_samples,))

    class TractablePotential(BasePotential):
        def __call__(self, theta, **kwargs):
            return target_distribution.log_prob(
                torch.as_tensor(theta, dtype=torch.float32)
            )

        def allow_iid_x(self) -> bool:
            return True

    potential_fn = TractablePotential(prior=MultivariateNormal(prior_mean, prior_cov))
    theta_transform = torch_tf.identity_transform

    posterior = VIPosterior(potential_fn=potential_fn, theta_transform=theta_transform)
    posterior.set_default_x(torch.tensor(np.zeros((num_dim,)).astype(np.float32)))
    posterior.vi_method = vi_method
    posterior.train(min_num_iters=500, learning_rate=1e-2)
    samples = posterior.sample((num_samples,), method=sampling_method)
    samples = torch.as_tensor(samples, dtype=torch.float32)

    check_c2st(samples, target_samples, alg="slice_np")


@pytest.mark.slow
@pytest.mark.parametrize("num_dim", (1, 2))
@pytest.mark.parametrize("q", ("maf", "nsf", "gaussian_diag", "gaussian", "mcf", "scf"))
def test_c2st_vi_flows_on_Gaussian(num_dim: int, q: str, set_seed):
    """Test VI on Gaussian, comparing to ground truth target via c2st.

    Args:
        num_dim: parameter dimension of the gaussian model
        vi_method: different vi methods
        sampling_method: Different sampling methods
        set_seed: fixture for manual seeding
    """
    # Coupling flows undefined at 1d
    if num_dim == 1 and q in ["mcf", "scf"]:
        return

    num_samples = 500

    likelihood_shift = -1.0 * ones(num_dim)
    likelihood_cov = 0.3 * eye(num_dim)
    prior_mean = zeros(num_dim)
    prior_cov = eye(num_dim)

    x_o = zeros((1, num_dim))
    target_distribution = true_posterior_linear_gaussian_mvn_prior(
        x_o[0], likelihood_shift, likelihood_cov, prior_mean, prior_cov
    )
    target_samples = target_distribution.sample((num_samples,))

    class TractablePotential(BasePotential):
        def __call__(self, theta, **kwargs):
            return target_distribution.log_prob(
                torch.as_tensor(theta, dtype=torch.float32)
            )

        def allow_iid_x(self) -> bool:
            return True

    potential_fn = TractablePotential(prior=MultivariateNormal(prior_mean, prior_cov))
    theta_transform = torch_tf.identity_transform

    posterior = VIPosterior(
        potential_fn=potential_fn, theta_transform=theta_transform, q=q
    )
    posterior.set_default_x(torch.tensor(np.zeros((num_dim,)).astype(np.float32)))
    posterior.train(min_num_iters=1000, learning_rate=1e-2, eps=1e-8)
    samples = posterior.sample((num_samples,))
    samples = torch.as_tensor(samples, dtype=torch.float32)

    check_c2st(samples, target_samples, alg="slice_np")


@pytest.mark.slow
@pytest.mark.parametrize("num_dim", (1, 2))
def test_c2st_vi_external_distributions_on_Gaussian(num_dim: int, set_seed):
    """Test VI on Gaussian, comparing to ground truth target via c2st.

    Args:
        num_dim: parameter dimension of the gaussian model
        vi_method: different vi methods
        sampling_method: Different sampling methods
        set_seed: fixture for manual seeding
    """
    num_samples = 500

    likelihood_shift = -1.0 * ones(num_dim)
    likelihood_cov = 0.3 * eye(num_dim)
    prior_mean = zeros(num_dim)
    prior_cov = eye(num_dim)

    x_o = zeros((1, num_dim))
    target_distribution = true_posterior_linear_gaussian_mvn_prior(
        x_o[0], likelihood_shift, likelihood_cov, prior_mean, prior_cov
    )
    target_samples = target_distribution.sample((num_samples,))

    class TractablePotential(BasePotential):
        def __call__(self, theta, **kwargs):
            return target_distribution.log_prob(
                torch.as_tensor(theta, dtype=torch.float32)
            )

        def allow_iid_x(self) -> bool:
            return True

    potential_fn = TractablePotential(prior=MultivariateNormal(prior_mean, prior_cov))
    theta_transform = torch_tf.identity_transform

    mu = zeros(num_dim, requires_grad=True)
    scale = torch.eye(num_dim, requires_grad=True)
    q = MultivariateNormal(mu, scale_tril=scale)

    posterior = VIPosterior(
        potential_fn=potential_fn,
        theta_transform=theta_transform,
        q=q,
        q_kwargs={"parameters": [mu, scale]},
    )
    posterior.set_default_x(torch.tensor(np.zeros((num_dim,)).astype(np.float32)))
    posterior.train(eps=1e-7)
    samples = posterior.sample((num_samples,))
    samples = torch.as_tensor(samples, dtype=torch.float32)

    check_c2st(samples, target_samples, alg="slice_np")


@pytest.mark.parametrize("q", ("maf", "nsf", "gaussian_diag", "gaussian", "mcf", "scf"))
def test_deepcopy_support(q: str, set_seed):
    """Tests if the variational does support deepcopy.

    Args:
        q: Different variational posteriors.
    """

    num_dim = 2

    class FakePotential(BasePotential):
        def __call__(self, theta, **kwargs):
            return torch.ones_like(torch.as_tensor(theta, dtype=torch.float32))

        def allow_iid_x(self) -> bool:
            return True

    potential_fn = FakePotential(prior=MultivariateNormal(zeros(num_dim), eye(num_dim)))
    theta_transform = torch_tf.identity_transform

    posterior = VIPosterior(
        potential_fn=potential_fn,
        theta_transform=theta_transform,
        q=q,
    )
    posterior_copy = deepcopy(posterior)
    posterior.set_default_x(torch.tensor(np.zeros((num_dim,)).astype(np.float32)))
    assert posterior._x != posterior_copy._x, "Mhh, something with the copy is strange"
    posterior_copy = deepcopy(posterior)
    assert (
        posterior._x == posterior_copy._x
    ).all(), "Mhh, something with the copy is strange"

    # Produces nonleaf tensors in the cache... -> Can lead to failure of deepcopy.
    posterior.q.rsample()
    posterior_copy = deepcopy(posterior)


@pytest.mark.slow
@pytest.mark.gpu
@pytest.mark.parametrize("num_dim", (1, 2))
@pytest.mark.parametrize("q", ("maf", "nsf", "gaussian_diag", "gaussian", "mcf", "scf"))
@pytest.mark.parametrize("vi_method", ("rKL", "fKL", "IW", "alpha"))
@pytest.mark.parametrize("sampling_method", ("naive", "sir"))
def test_vi_on_gpu(
    num_dim: int, q: Distribution, vi_method: str, sampling_method: str, set_seed
):
    """Test VI on Gaussian, comparing to ground truth target via c2st.

    Args:
        num_dim: parameter dimension of the gaussian model
        vi_method: different vi methods
        sampling_method: Different sampling methods
        set_seed: fixture for manual seeding
    """

    device = "cuda:0"

    if num_dim == 1 and q in ["mcf", "scf"]:
        return

    # Good run where everythink is one the correct device.
    class FakePotential(BasePotential):
        def __call__(self, theta, **kwargs):
            return torch.ones(len(theta), dtype=torch.float32, device=device)

        def allow_iid_x(self) -> bool:
            return True

    potential_fn = FakePotential(
        prior=MultivariateNormal(
            zeros(num_dim, device=device), eye(num_dim, device=device)
        ),
        device=device,
    )
    theta_transform = torch_tf.identity_transform

    posterior = VIPosterior(
        potential_fn=potential_fn, theta_transform=theta_transform, q=q, device=device
    )
    posterior.set_default_x(
        torch.tensor(np.zeros((num_dim,)).astype(np.float32)).to(device)
    )
    posterior.vi_method = vi_method

    samples = posterior.sample()
    logprobs = posterior.log_prob(samples)

    print(samples)
    assert str(samples.device) == device, "The devices does not match"
    assert str(logprobs.device) == device, "The devices does not match"

    posterior.train(min_num_iters=9, max_num_iters=10, warm_up_rounds=10)
    samples = posterior.sample((1,), method=sampling_method)
    logprobs = posterior.log_prob(samples)

    assert str(samples.device) == device, "The devices after training does not match"
    assert str(logprobs.device) == device, "The devices after training does not match"
