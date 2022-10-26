import sbibm
import torch
import pickle
import time

from sbibm.metrics.c2st import c2st

from sbi.inference import SNPE, prepare_for_sbi, simulate_for_sbi
from sbi import utils

# Get simulator

slcp2 = sbibm.get_task("slcp")  # See sbibm.get_available_tasks() for all tasks
slcp_simulator2 = slcp2.get_simulator()
slcp_observation2 = slcp2.get_observation(num_observation=1)  # 10 per task
prior_min = [-3] * 5
prior_max = [3] * 5
slcp_prior_2 = utils.torchutils.BoxUniform(
    low=torch.as_tensor(prior_min), high=torch.as_tensor(prior_max)
)

num_rounds = 20
simulator, prior = prepare_for_sbi(slcp_simulator2, slcp_prior_2)
acc_list = []
for i in range(1, 11):


    reference_samples = slcp2.get_reference_posterior_samples(num_observation=i)
    ref_obs = slcp2.get_observation(num_observation=i)
    simulator, prior = prepare_for_sbi(slcp_simulator2, slcp_prior_2)
    proposal = prior
    start_time = time.time()
    for j in range(num_rounds):
        theta, x = simulate_for_sbi(simulator, proposal, num_simulations=1000, num_workers=7)
        if j == 0: # Doing nothing?
            embedding_net = torch.nn.Identity()
            neural_posterior = utils.posterior_nn(model="maf", embedding_net=embedding_net)
            inference = SNPE(prior=prior, density_estimator=neural_posterior)
        density_estimator = inference.append_simulations(theta, x, proposal=proposal).train()
        posterior = inference.build_posterior(density_estimator)
        proposal = posterior.set_default_x(ref_obs)
    time_taken = time.time() - start_time
    #
    with open(f'relu_test_data/obs_{i}_posterior_full.pkl','wb') as f:
        pickle.dump(posterior, f)

    samples, acceptance_rate = posterior.sample((10000,), return_acceptance_rate=True)
    c2st_accuracy = c2st(samples, reference_samples)

    with open('relu_test_data/manual_logging.txt', 'a') as f:
        f.write(f'c2st_full_accuracy_obs_{i} = {c2st_accuracy}, time_taken = {time_taken}, acceptance_rate = {acceptance_rate}\n')


    proposal = prior
    start_time = time.time()
    for j in range(num_rounds):
        theta, x = simulate_for_sbi(simulator, proposal, num_simulations=1000, num_workers=7)
        if j == 0: # Doing nothing?
            embedding_net = torch.nn.Identity()
            neural_posterior = utils.posterior_nn(model="maf", embedding_net=embedding_net)
            inference = SNPE(prior=prior, density_estimator=neural_posterior)
        density_estimator = inference.append_simulations(theta, x, proposal=proposal).train(loss_function='leakage_free')
        posterior = inference.build_posterior(density_estimator)
        proposal = posterior.set_default_x(ref_obs)
    time_taken = time.time() - start_time


    # with open(f'relu_test_data/obs_{i}_posterior_relu.pkl','wb') as f:
    #     pickle.dump(posterior, f)

    samples, acceptance_rate = posterior.sample((10000,), return_acceptance_rate=True)
    c2st_accuracy = c2st(samples, reference_samples)

    with open('relu_test_data/manual_logging.txt', 'a') as f:
        f.write(f'c2st_noleak_accuracy_obs_{i} = {c2st_accuracy}, time_taken = {time_taken}, acceptance_rate = {acceptance_rate}\n')
