import numpy as np
import torch
import pickle
import sys
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
from torch.nn.functional import normalize

# remove top and right axis from plots
mpl.rcParams["axes.spines.right"] = False
mpl.rcParams["axes.spines.top"] = False
prior_min = [0.5, 1e-4, 1e-4, 1e-4, 50, 40, 1e-4, 35]
prior_max = [80.0, 15.0, 0.6, 0.6, 3000, 90, 0.15, 100]
prior = utils.torchutils.BoxUniform(
    low=torch.as_tensor(prior_min), high=torch.as_tensor(prior_max)
)
## Simulator
from HH_Helpers import syn_current, calculate_summary_statistics

I, t_on, t_off, dt, t, A_soma = syn_current()

from HH_Helpers import HHsimulator


def run_HH_model(params):

    params = np.asarray(params)

    # input current, time step
    I, t_on, t_off, dt, t, A_soma = syn_current()

    t = np.arange(0, len(I), 1) * dt

    # initial voltage
    V0 = -70

    states = HHsimulator(V0, params.reshape(1, -1), dt, t, I)

    return dict(data=states.reshape(-1), time=t, dt=dt, I=I.reshape(-1))


def summ_simulation_wrapper(params):
    """
    Returns summary statistics from conductance values in `params`.
    
    Summarizes the output of the HH simulator and converts it to `torch.Tensor`.
    """
    obs = run_HH_model(params)
    summstats = torch.as_tensor(calculate_summary_statistics(obs))
    return summstats


def full_simulation_wrapper(params):
    """
    Returns summary statistics from conductance values in `params`.
    
    Summarizes the output of the HH simulator and converts it to `torch.Tensor`.
    """
    obs = torch.tensor(run_HH_model(params)["data"])
    return obs


## Distances
def hausdorff_dist(x, y, reverse=False):
    x_pairs = np.array([x["time"], x["data"]]).T
    y_pairs = np.array([y["time"], y["data"]]).T
    if reverse:
        return directed_hausdorff(y_pairs, x_pairs)
    return directed_hausdorff(x_pairs, y_pairs)
    # x_pairs = torch.vstack([x['time'],x['data']])


## Simulator
true_params = prior.sample()
observation_trace = run_HH_model(true_params)

grid_simulator, prior = prepare_for_sbi(full_simulation_wrapper, prior)
theta, data = simulate_for_sbi(
    grid_simulator, prior, num_simulations=100, num_workers=7
)

data_np = data.detach().numpy()

time = observation_trace["time"]


def get_hausdorff_simulator(data):
    def hausdorff_simulator(params, input=None):
        if input is None:
            input = run_HH_model(params)
        distances = []
        for array in data:
            aug_data = {"time": time, "data": array}
            distances.append(torch.tensor(hausdorff_dist(input, aug_data)[0]))
        return torch.tensor(distances)

    return hausdorff_simulator


hausdorff_simulator = get_hausdorff_simulator(data_np)
points_ref = hausdorff_simulator(true_params, observation_trace)
summs_ref = calculate_summary_statistics(observation_trace)

posterior = infer(
    hausdorff_simulator, prior, method="SNPE", num_simulations=100, num_workers=7
)
list(zip(prior_min, prior_max))

posterior_summ = infer(
    summ_simulation_wrapper, prior, method="SNPE", num_simulations=100, num_workers=7
)


with open("test_posterior_2", "wb") as f:
    pickle.dump(posterior, f)

