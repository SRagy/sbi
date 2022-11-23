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
for i in range(2, 11):
    reference_samples = slcp2.get_reference_posterior_samples(num_observation=i)
    ref_obs = slcp2.get_observation(num_observation=i)
    simulator, prior = prepare_for_sbi(slcp_simulator2, slcp_prior_2)
    proposal = prior
    start_time = time.time()
    inference = SNPE(prior=prior)
    acceptance_rate = 1
    for j in range(num_rounds):
        print(f"round {j}")
        theta, x = simulate_for_sbi(simulator, proposal, num_simulations=1000)
        density_estimator = inference.append_simulations(theta, x, proposal=proposal).train(loss_function='automatic',
                                                                                            num_norm_samples=1,
                                                                                            stop_after_epochs=20,
                                                                                            leak_correction_frequency=0.1)
        posterior = inference.build_posterior(density_estimator)
        proposal = posterior.set_default_x(ref_obs)
        samples, acceptance_rate = posterior.sample((10000,), return_acceptance_rate=True)

    time_taken = time.time() - start_time
    c2st_accuracy = c2st(samples, reference_samples)

    with open('Leak_Fix/manual_logging.txt', 'a') as f:
        f.write(f'automatic_detached_obs_{i} = {c2st_accuracy}, time_taken = {time_taken}, acceptance_rate = {acceptance_rate}\n')

