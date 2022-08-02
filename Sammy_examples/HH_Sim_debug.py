## Preliminaries
import numpy as np
import torch
from tqdm import tqdm
from joblib import Parallel, delayed
from pyro.distributions.empirical import Empirical

# visualization
import matplotlib as mpl
import matplotlib.pyplot as plt

# sbi
from sbi.inference.base import infer
from sbi.inference import SNPE, prepare_for_sbi, simulate_for_sbi
from sbi.utils.get_nn_models import posterior_nn
from sbi import utils
from sbi import analysis

# distances
from scipy.spatial.distance import directed_hausdorff

# remove top and right axis from plots
mpl.rcParams["axes.spines.right"] = False
mpl.rcParams["axes.spines.top"] = False
# Simulator
from HH_helper_functions import syn_current

I, t_on, t_off, dt, t, A_soma = syn_current()
from HH_helper_functions import HHsimulator


def run_HH_model(params):

    params = np.asarray(params)

    # input current, time step
    I, t_on, t_off, dt, t, A_soma = syn_current()

    t = np.arange(0, len(I), 1) * dt

    # initial voltage
    V0 = -70

    states = HHsimulator(V0, params.reshape(1, -1), dt, t, I)

    return dict(data=states.reshape(-1), time=t, dt=dt, I=I.reshape(-1))


# three sets of (g_Na, g_K)
params = np.array([[50.0, 1.0], [4.0, 1.5], [20.0, 15.0]])

num_samples = len(params[:, 0])
sim_samples = np.zeros((num_samples, len(I)))
for i in range(num_samples):
    sim_samples[i, :] = run_HH_model(params=params[i, :])["data"]
# colors for traces
col_min = 2
num_colors = num_samples + col_min
cm1 = mpl.cm.Blues
col1 = [cm1(1.0 * i / num_colors) for i in range(col_min, num_colors)]

fig = plt.figure(figsize=(7, 5))
gs = mpl.gridspec.GridSpec(2, 1, height_ratios=[4, 1])
ax = plt.subplot(gs[0])
for i in range(num_samples):
    plt.plot(t, sim_samples[i, :], color=col1[i], lw=2)
plt.ylabel("voltage (mV)")
ax.set_xticks([])
ax.set_yticks([-80, -20, 40])

ax = plt.subplot(gs[1])
plt.plot(t, I * A_soma * 1e3, "k", lw=2)
plt.xlabel("time (ms)")
plt.ylabel("input (nA)")

ax.set_xticks([0, max(t) / 2, max(t)])
ax.set_yticks([0, 1.1 * np.max(I * A_soma * 1e3)])
ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%.2f"))
plt.show()
from HH_helper_functions import calculate_summary_statistics


def simulation_wrapper(params):
    """
    Returns summary statistics from conductance values in `params`.
    
    Summarizes the output of the HH simulator and converts it to `torch.Tensor`.
    """
    obs = run_HH_model(params)
    summstats = torch.as_tensor(calculate_summary_statistics(obs))
    return summstats


true_params = np.array([50.0, 5.0])
labels_params = [r"$g_{Na}$", r"$g_{K}$"]
observation_trace = run_HH_model(true_params)
observation_summary_statistics = calculate_summary_statistics(observation_trace)
prior_min = [0.5, 1e-4]
prior_max = [80.0, 15.0]
prior = utils.torchutils.BoxUniform(
    low=torch.as_tensor(prior_min), high=torch.as_tensor(prior_max)
)
## Distance functions
def euclidean(x, y):
    return np.linalg.norm(x - y)


def simulation_wrapper_euclidean(params):
    """
    Uses euclidean distances from a given observation as summary
    """
    data = observation_trace["data"]
    obs = run_HH_model(params)["data"]
    summstats = euclidean(data, obs)
    return torch.as_tensor([summstats])


def hausdorff_dist(x, y, reverse=False):
    x_pairs = np.array([x["time"], x["data"]]).T
    y_pairs = np.array([y["time"], y["data"]]).T
    if reverse:
        return directed_hausdorff(y_pairs, x_pairs)
    return directed_hausdorff(x_pairs, y_pairs)
    # x_pairs = torch.vstack([x['time'],x['data']])


def simulation_wrapper_hausdorff(params, reverse=False):
    data = observation_trace
    obs = run_HH_model(params)
    summstats = hausdorff_dist(data, obs, reverse)
    return torch.as_tensor([summstats[0]])


# Create categorical distributions from proposals
def effective_sample_size(w):
    """Effective sample size of weights

    `w` is a 1-dimensional tensor of weights (normalised or unnormalised)"""
    sumw = torch.sum(w)
    if sumw == 0:
        return 0.0

    return (sumw ** 2.0) / torch.sum(w ** 2.0)


def get_alternate_weights(sqd, old_weights, eps):
    """Return weights appropriate to another `epsilon` value"""
    # Interpretable version of the generic reweighting code:
    # w = old_weights
    # d = torch.sqrt(sqd)
    # w /= torch.exp(-0.5*(d / old_eps)**2.)
    # w *= torch.exp(-0.5*(d / new_eps)**2.)
    # w /= sum(w)

    w = old_weights.detach().clone()
    if eps == 0:
        # Remove existing distance-based weight contribution
        # Replace with indicator function weight contribution
        w = torch.where(sqd == 0.0, w, torch.zeros_like(w))
    else:
        # An efficient way to do the generic case
        a = -0.5 * eps ** -2.0
        w *= torch.exp(sqd * a)

    sumw = torch.sum(w)
    if sumw > 0.0:
        w /= sumw
    return w


def find_eps(sqd, old_weights, target_ess, upper, bisection_its=50):
    """Return epsilon value <= `upper` giving ess matching `target_ess` as closely as possible

        Bisection search is performed using `bisection_its` iterations
        """
    w = get_alternate_weights(sqd, old_weights, upper)
    ess = effective_sample_size(w)
    if ess < target_ess:
        return upper

    lower = 0.0
    for i in range(bisection_its):
        eps_guess = (lower + upper) / 2.0
        w = get_alternate_weights(sqd, old_weights, eps_guess)
        ess = effective_sample_size(w)
        if ess > target_ess:
            upper = eps_guess
        else:
            lower = eps_guess

    # Consider returning eps=0 if it's still an endpoint
    if lower == 0.0:
        w = get_alternate_weights(sqd, old_weights, 0.0)
        ess = effective_sample_size(w)
        if ess > target_ess:
            return 0.0

    # Be conservative by returning upper end of remaining range
    return upper


class Empirical_Fix(Empirical):
    def __init__(self, samples, log_weights, validate_args=None):
        super().__init__(samples, log_weights, validate_args=validate_args)

    def log_prob(self, value):
        sample_shape = super().sample().shape
        sample_shape_len = len(sample_shape)
        if value.shape == sample_shape:
            return super().log_prob(value)

        if sample_shape_len:
            assert (
                value.shape[-sample_shape_len:] == sample_shape
            ), "The value needs to consist of samples from the empricial distribution"
            batch_shape = value.shape[:-sample_shape_len]
        else:
            batch_shape = value.shape
        num_samples = batch_shape.numel()

        value_reshape = value.reshape((num_samples,) + sample_shape)
        log_probs = []

        for i in value_reshape:
            log_probs.append(super().log_prob(i))

        log_probs_tensor = torch.tensor(log_probs)
        return log_probs_tensor.reshape(batch_shape)


def to_empirical(
    simulator,
    distribution,
    distance_function,
    num_simulations,
    num_workers,
    obs,
    old_epsilon=np.inf,
):
    theta, x = simulate_for_sbi(
        simulator,
        distribution,
        num_simulations=num_simulations,
        num_workers=num_workers,
    )
    distances = Parallel(n_jobs=num_workers, verbose=10)(
        delayed(distance_function)(data, obs) for data in x
    )
    sqd = torch.tensor(distances) ** 2
    old_log_prob = distribution.log_prob(theta)
    old_weights = np.exp(old_log_prob) / np.exp(old_log_prob).sum()
    if old_epsilon == np.inf:
        upper = torch.max(sqd).item()
    else:
        upper = old_epsilon
    epsilon = find_eps(sqd, old_weights, target_ess=num_simulations / 10, upper=upper)
    log_prob = old_log_prob - 0.5 * sqd / epsilon ** 2
    return epsilon, Empirical_Fix(theta, log_prob)


# Build posterior

obs = observation_trace["data"]
simulator, prior = prepare_for_sbi(simulation_wrapper_euclidean, prior)
epsilon, new_prior = to_empirical(
    simulator, prior, euclidean, 100, 7, obs, old_epsilon=np.inf
)
simulator, new_prior = prepare_for_sbi(simulation_wrapper_euclidean, new_prior)
inference = SNPE(prior=prior)
proposal = new_prior
theta, x = simulate_for_sbi(simulator, proposal, num_simulations=100, num_workers=7)
density_estimator = inference.append_simulations(
    theta, x, proposal=new_prior
).train()  # Will this work with empirical prop
posterior = inference.build_posterior(density_estimator)

