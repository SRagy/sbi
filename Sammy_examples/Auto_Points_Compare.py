import sbibm
import numpy as np
import torch
import pickle
import pyro
from tqdm import tqdm
from joblib import Parallel, delayed
from pyro.distributions.empirical import Empirical
from functools import reduce
from scipy.optimize import linear_sum_assignment
import timeit
from sbibm.metrics.c2st import c2st


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

sbibm.get_available_tasks()


# Define averaging by minimum matching


def minimum_matching(x_old, x_new, epsilon):
    if x_old is None:
        return x_new
    weights = torch.cdist(x_old, x_new)
    row_ind, col_ind = linear_sum_assignment(weights)
    return (1 - epsilon) * x_old + epsilon * x_new[col_ind]


# Get simulator

slcp2 = sbibm.get_task("slcp")  # See sbibm.get_available_tasks() for all tasks
slcp_simulator2 = slcp2.get_simulator()
slcp_observation2 = slcp2.get_observation(num_observation=1)  # 10 per task
prior_min = [-3] * 5
prior_max = [3] * 5
slcp_prior_2 = utils.torchutils.BoxUniform(
    low=torch.as_tensor(prior_min), high=torch.as_tensor(prior_max)
)


def get_slcp_simulator(points, seed=None):
    def simulator(theta, input=None, seed=seed):
        if seed is not None:
            torch.manual_seed(seed)
            # np.random.seed(seed)
            # pyro.set_rng_seed(seed)
        if input is None:
            full_data = slcp_simulator2(theta)
        else:
            full_data = input
        return euclidean_sq(points, full_data)

    return simulator

thetas = slcp_prior_2.sample((32,))
points = slcp_simulator2(thetas)
rand_simulator = get_slcp_simulator(points)
# Define several classes for various kinds of learnable point updates


class BestMatching(torch.nn.Module):
    """
    Learns an update rule by sampling from proposal and performing minimum matching.
    It then averages existing and old points by an amount determined by learnable parameter.
    We consider this eps = cos^2(theta).

    """

    def __init__(self, old_points, new_points):
        """
        Instantiate parameter to be used as weighting.
        """
        super().__init__()
        self.theta = torch.nn.Parameter(torch.tensor(torch.pi / 2))
        self.old_points = old_points
        self.new_points = new_points

    def forward(self, x):
        # (1-eps) * old_points + eps * new_points
        # point_update = old_points + (torch.cos(self.theta)**2)*(self.new_points - self.old_points)
        point_update = minimum_matching(
            self.old_points, self.new_points, torch.cos(self.theta) ** 2
        )  # move to init then reinitialise?
        distances = torch.cdist(x, point_update)

        return distances


class BestMatching2(torch.nn.Module):
    def __init__(self, old_points, new_points):
        """
        Instantiate parameter to be used as weighting.
        """
        super().__init__()
        num_params = len(new_points)

        self.thetas = torch.nn.Parameter(torch.full((num_params, 1), torch.pi / 2))
        self.old_points = old_points
        self.new_points = new_points

    def forward(self, x):
        # (1-eps) * old_points + eps * new_points
        # point_update = old_points + (torch.cos(self.theta)**2)*(self.new_points - self.old_points)
        point_update = minimum_matching(
            self.old_points, self.new_points, torch.cos(self.thetas) ** 2
        )  # move to init then reinitialise?
        distances = torch.cdist(x, point_update)

        return distances


class BestMatchingSig(torch.nn.Module):
    """
    Learns an update rule by sampling from proposal and performing minimum matching.
    It then averages existing and old points by an amount determined by learnable parameter.
    We consider this eps = sigmoid(theta).

    """

    def __init__(self, old_points, new_points):
        """
        Instantiate parameter to be used as weighting.
        """
        super().__init__()
        self.theta = torch.nn.Parameter(torch.tensor(-5.0))
        self.old_points = old_points
        self.new_points = new_points

    def forward(self, x):
        # (1-eps) * old_points + eps * new_points
        # point_update = old_points + (torch.cos(self.theta)**2)*(self.new_points - self.old_points)
        point_update = minimum_matching(
            self.old_points, self.new_points, torch.sigmoid(self.theta) ** 2
        )  # move to init then reinitialise?
        distances = torch.cdist(x, point_update)

        return distances


class AllPoints(torch.nn.Module):
    def __init__(self, init_points):
        """
        Instantiate parameter to be used as weighting.
        """
        super().__init__()
        # self.linear_layer = torch.nn.Linear(8, num_points)
        # self._matrix = torch.nn.Parameter(torch.randn((8,num_points)))
        self._matrix = torch.nn.Parameter(init_points)
        self._initial_basis = torch.eye(8)

    def forward(self, x):
        # (1-eps) * old_points + eps * new_points
        # point_update = old_points + (torch.cos(self.theta)**2)*(self.new_points - self.old_points)
        # points_to_compare = self.linear_layer(self._initial_basis).T
        points_to_compare = self._matrix @ self._initial_basis

        # essentially we're actually just interested in weight matrix
        distances = torch.cdist(x, points_to_compare)

        return distances


num_rounds = 10
simulator, prior = prepare_for_sbi(slcp_simulator2, slcp_prior_2)
acc_list = []
for i in range(1, 11):
    reference_samples = slcp2.get_reference_posterior_samples(num_observation=i)
    ref_obs = slcp2.get_observation(num_observation=i)
    simulator, prior = prepare_for_sbi(slcp_simulator2, slcp_prior_2)
    proposal = prior
    old_points = None
    # neural_posterior = utils.posterior_nn(model="maf", embedding_net=torch.nn.Identity())
    # inference = SNPE(prior=prior, density_estimator=neural_posterior)
    for j in range(num_rounds):
        theta, x = simulate_for_sbi(simulator, proposal, num_simulations=1000, num_workers=7)
        if j == 0: # Doing nothing?
            embedding_net = BestMatching(old_points, x[:32])
            neural_posterior = utils.posterior_nn(model="maf", embedding_net=embedding_net)
            inference = SNPE(prior=prior, density_estimator=neural_posterior)
        else:
            density_estimator._embedding_net = BestMatching(old_points, x[:32])
            old_points = x[:32]
        density_estimator = inference.append_simulations(theta, x, proposal=proposal).train()
        posterior = inference.build_posterior(density_estimator)
        proposal = posterior.set_default_x(ref_obs)

    with open(f'auto_points_data_2/obs_{i}_posterior_bm.pkl','wb') as f:
        pickle.dump(posterior, f)

    samples = posterior.sample((10000,))
    c2st_accuracy = c2st(samples, reference_samples)

    with open('auto_points_data_2/manual_logging.txt', 'a') as f:
        f.write(f'c2st_bm_accuracy_obs_{i} = {c2st_accuracy}\n')

    simulator, prior = prepare_for_sbi(slcp_simulator2, slcp_prior_2)
    proposal = prior
    for j in range(num_rounds):
        theta, x = simulate_for_sbi(simulator, proposal, num_simulations=1000, num_workers=7)
        if j == 0: # Doing nothing?
            embedding_net = torch.nn.Identity()
            neural_posterior = utils.posterior_nn(model="maf", embedding_net=embedding_net)
            inference = SNPE(prior=prior, density_estimator=neural_posterior)
        density_estimator = inference.append_simulations(theta, x, proposal=proposal).train()
        posterior = inference.build_posterior(density_estimator)
        proposal = posterior.set_default_x(ref_obs)


    with open(f'auto_points_data_2/obs_{i}_posterior_full.pkl','wb') as f:
        pickle.dump(posterior, f)

    samples = posterior.sample((10000,))
    c2st_accuracy = c2st(samples, reference_samples)

    with open('auto_points_data_2/manual_logging.txt', 'a') as f:
        f.write(f'c2st_full_accuracy_obs_{i} = {c2st_accuracy}\n')

    simulator, prior = prepare_for_sbi(slcp_simulator2, slcp_prior_2)
    proposal = prior
    for j in range(num_rounds):
        theta, x = simulate_for_sbi(simulator, proposal, num_simulations=1000, num_workers=7)
        if j == 0:
            embedding_net = AllPoints(x[:32])
            neural_posterior = utils.posterior_nn(model="maf", embedding_net=embedding_net)
            inference = SNPE(prior=prior, density_estimator=neural_posterior)
        density_estimator = inference.append_simulations(theta, x, proposal=proposal).train()
        posterior = inference.build_posterior(density_estimator)
        proposal = posterior.set_default_x(ref_obs)

    with open(f'auto_points_data_2/obs_{i}_posterior_allpts.pkl','wb') as f:
        pickle.dump(posterior, f)

    samples = posterior.sample((10000,))
    c2st_accuracy = c2st(samples, reference_samples)

    with open('auto_points_data_2/manual_logging.txt', 'a') as f:
        f.write(f'c2st_allpts_accuracy_obs_{i} = {c2st_accuracy}\n')


    simulator, prior = prepare_for_sbi(slcp_simulator2, slcp_prior_2)
    proposal = prior
    embedding_net = torch.nn.Linear(8,32)
    neural_posterior = utils.posterior_nn(model="maf", embedding_net=embedding_net)
    inference = SNPE(prior=prior, density_estimator=neural_posterior)
    for j in range(num_rounds):
        theta, x = simulate_for_sbi(simulator, proposal, num_simulations=1000, num_workers=7)
        density_estimator = inference.append_simulations(theta, x, proposal=proposal).train()
        posterior = inference.build_posterior(density_estimator)
        proposal = posterior.set_default_x(ref_obs)

    with open(f'auto_points_data_2/obs_{i}_posterior_proj.pkl','wb') as f:
        pickle.dump(posterior, f)

    samples = posterior.sample((10000,))
    c2st_accuracy = c2st(samples, reference_samples)

    with open('auto_points_data_2/manual_logging.txt', 'a') as f:
        f.write(f'c2st_proj_accuracy_obs_{i} = {c2st_accuracy}\n')


    ref_point = rand_simulator(None, ref_obs)
    simulator, prior = prepare_for_sbi(rand_simulator, slcp_prior_2)
    proposal = prior
    neural_posterior = utils.posterior_nn(model="maf", embedding_net=torch.nn.Identity())
    inference = SNPE(prior=prior, density_estimator=neural_posterior)
    for j in range(num_rounds):
        theta, x = simulate_for_sbi(simulator, proposal, num_simulations=1000, num_workers=7)
        density_estimator = inference.append_simulations(theta, x, proposal=proposal).train()
        posterior = inference.build_posterior(density_estimator)
        proposal = posterior.set_default_x(ref_point)

    with open(f'auto_points_data_2/obs_{i}_posterior_fixed.pkl','wb') as f:
        pickle.dump(posterior, f)

    samples = posterior.sample((10000,))
    c2st_accuracy = c2st(samples, reference_samples)

    with open('auto_points_data_2/manual_logging.txt', 'a') as f:
        f.write(f'c2st_fixed_accuracy_obs_{i} = {c2st_accuracy}\n')