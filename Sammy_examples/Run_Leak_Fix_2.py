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
slcp_prior_2 = slcp2.get_prior_dist()

num_rounds = 10
simulator, prior = prepare_for_sbi(slcp_simulator2, slcp_prior_2)
acc_list = []
for i in range(1, 11):
    reference_samples = slcp2.get_reference_posterior_samples(num_observation=i)
    ref_obs = slcp2.get_observation(num_observation=i)
    simulator, prior = prepare_for_sbi(slcp_simulator2, slcp_prior_2)
    proposal = prior
    start_time = time.time()
    inference = SNPE(prior=prior)
    for j in range(num_rounds):
        theta, x = simulate_for_sbi(simulator, proposal, num_simulations=1000)
        if j > 1:
            density_estimator = inference.train(loss_function='correction', num_norm_samples=5, training_batch_size=100,
                                                stop_after_epochs=50)
        density_estimator = inference.append_simulations(theta, x, proposal=proposal).train(loss_function='default',
                                                                                            num_norm_samples=1)
        posterior = inference.build_posterior(density_estimator)
        proposal = posterior.set_default_x(ref_obs)

    time_taken = time.time() - start_time

    samples, acceptance_rate = posterior.sample((10000,), return_acceptance_rate=True)
    c2st_accuracy = c2st(samples, reference_samples)

    with open('Leak_Fix/manual_logging.txt', 'a') as f:
        f.write(f'c2st_fix2_accuracy_obs_{i} = {c2st_accuracy}, time_taken = {time_taken}, acceptance_rate = {acceptance_rate}\n')

