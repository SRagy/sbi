import os
from copy import deepcopy

import numpy as np
import sbi.utils as utils
import torch
from sbi.utils.torchutils import get_default_device
from pyro.infer.mcmc import HMC, NUTS
from pyro.infer.mcmc.api import MCMC
from torch import distributions
from torch import optim
from torch.nn.utils import clip_grad_norm_
from torch.utils import data
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sbi.inference.posteriors.sbi_posterior import Posterior
from typing import Optional

import sbi.simulators as simulators
import sbi.utils as utils
from sbi.mcmc import Slice, SliceSampler
from sbi.simulators.simutils import set_simulator_attributes
from sbi.utils.torchutils import get_default_device


class SNL:
    """
    Implementation of
    'Sequential Neural Likelihood: Fast Likelihood-free Inference with Autoregressive Flows'
    Papamakarios et al.
    AISTATS 2019
    https://arxiv.org/abs/1805.07226
    """

    def __init__(
        self,
        simulator,
        prior: torch.distributions,
        true_observation: torch.Tensor,
        density_estimator: Optional[torch.nn.Module],
        summary_writer: SummaryWriter = None,
        device: torch.device = None,
        mcmc_method: str = "slice-np",
    ):
        """
        :param simulator: Python object with 'simulate' method which takes a torch.Tensor
        of parameter values, and returns a simulation result for each parameter as a torch.Tensor.
        :param prior: Distribution object with 'log_prob' and 'sample' methods.
        :param true_observation: torch.Tensor containing the observation x0 for which to
        perform inference on the posterior p(theta | x0).
        :param neural_likelihood: Conditional density estimator q(x | theta) in the form of an
        nets.Module. Must have 'log_prob' and 'sample' methods.
        :param mcmc_method: MCMC method to use for posterior sampling. Must be one of
        ['slice', 'hmc', 'nuts'].
        :param summary_writer: SummaryWriter
            Optionally pass summary writer. A way to change the log file location.
            If None, will create one internally, saving logs to cwd/logs.
        :param device: torch.device
            Optionally pass device
            If None, will infer it
        """

        # set name and dimensions of simulator
        simulator = set_simulator_attributes(simulator, prior)

        self._simulator = simulator
        self._prior = prior
        self._true_observation = true_observation
        self._device = get_default_device() if device is None else device

        # create the deep neural density estimator
        if density_estimator is None:
            density_estimator = utils.likelihood_nn(
                model="maf", prior=self._prior, context=self._true_observation,
            )

        # create neural posterior which can sample()
        self._neural_posterior = Posterior(
            algorithm="snl",
            neural_net=density_estimator,
            prior=prior,
            context=true_observation,
            mcmc_method=mcmc_method,
        )

        # switch to training mode
        self._neural_posterior.neural_net.train()

        # Need somewhere to store (parameter, observation) pairs from each round.
        self._parameter_bank, self._observation_bank = [], []

        # Each SNL run has an associated log directory for TensorBoard output.
        if summary_writer is None:
            log_dir = os.path.join(
                utils.get_log_root(), "snl", simulator.name, utils.get_timestamp()
            )
            self._summary_writer = SummaryWriter(log_dir)
        else:
            self._summary_writer = summary_writer

        # Each run also has a dictionary of summary statistics which are populated
        # over the course of training.
        self._summary = {
            "mmds": [],
            "median-observation-distances": [],
            "negative-log-probs-true-parameters": [],
            "neural-net-fit-times": [],
            "mcmc-times": [],
            "epochs": [],
            "best-validation-log-probs": [],
        }

    def run_inference(self, num_rounds, num_simulations_per_round):
        """
        This runs SNL for num_rounds rounds, using num_simulations_per_round calls to
        the simulator per round.

        :param num_rounds: Number of rounds to run.
        :param num_simulations_per_round: Number of simulator calls per round.
        :return: None
        """

        round_description = ""
        tbar = tqdm(range(num_rounds))
        for round_ in tbar:

            tbar.set_description(round_description)

            # Generate parameters from prior in first round, and from most recent posterior
            # estimate in subsequent rounds.
            if round_ == 0:
                parameters, observations = simulators.simulation_wrapper(
                    simulator=self._simulator,
                    parameter_sample_fn=lambda num_samples: self._prior.sample(
                        (num_samples,)
                    ),
                    num_samples=num_simulations_per_round,
                )
            else:
                parameters, observations = simulators.simulation_wrapper(
                    simulator=self._simulator,
                    parameter_sample_fn=lambda num_samples: self._neural_posterior.sample(
                        num_samples
                    ),
                    num_samples=num_simulations_per_round,
                )

            # Store (parameter, observation) pairs.
            self._parameter_bank.append(torch.Tensor(parameters))
            self._observation_bank.append(torch.Tensor(observations))

            # Fit neural likelihood to newly aggregated dataset.
            self._fit_likelihood()

            # Update description for progress bar.
            round_description = (
                f"-------------------------\n"
                f"||||| ROUND {round_ + 1} STATS |||||:\n"
                f"-------------------------\n"
                f"Epochs trained: {self._summary['epochs'][-1]}\n"
                f"Best validation performance: {self._summary['best-validation-log-probs'][-1]:.4f}\n\n"
            )

            # Update TensorBoard and summary dict.
            self._summary_writer, self._summary = utils.summarize(
                summary_writer=self._summary_writer,
                summary=self._summary,
                round_=round_,
                true_observation=self._true_observation,
                parameter_bank=self._parameter_bank,
                observation_bank=self._observation_bank,
                simulator=self._simulator,
            )
        return self._neural_posterior

    def _fit_likelihood(
        self,
        batch_size=100,
        learning_rate=5e-4,
        validation_fraction=0.1,
        stop_after_epochs=20,
    ):
        """
        Trains the conditional density estimator for the likelihood by maximum likelihood
        on the most recently aggregated bank of (parameter, observation) pairs.
        Uses early stopping on a held-out validation set as a terminating condition.

        :param batch_size: Size of batch to use for training.
        :param learning_rate: Learning rate for Adam optimizer.
        :param validation_fraction: The fraction of data to use for validation.
        :param stop_after_epochs: The number of epochs to wait for improvement on the
        validation set before terminating training.
        :return: None
        """

        # Get total number of training examples.
        num_examples = torch.cat(self._parameter_bank).shape[0]

        # Select random train and validation splits from (parameter, observation) pairs.
        permuted_indices = torch.randperm(num_examples)
        num_training_examples = int((1 - validation_fraction) * num_examples)
        num_validation_examples = num_examples - num_training_examples
        train_indices, val_indices = (
            permuted_indices[:num_training_examples],
            permuted_indices[num_training_examples:],
        )

        # Dataset is shared for training and validation loaders.
        dataset = data.TensorDataset(
            torch.cat(self._observation_bank), torch.cat(self._parameter_bank)
        )

        # Create neural_net and validation loaders using a subset sampler.
        train_loader = data.DataLoader(
            dataset,
            batch_size=batch_size,
            drop_last=True,
            sampler=SubsetRandomSampler(train_indices),
        )
        val_loader = data.DataLoader(
            dataset,
            batch_size=min(batch_size, num_examples - num_training_examples),
            shuffle=False,
            drop_last=False,
            sampler=SubsetRandomSampler(val_indices),
        )

        optimizer = optim.Adam(
            self._neural_posterior.neural_net.parameters(), lr=learning_rate
        )
        # Keep track of best_validation log_prob seen so far.
        best_validation_log_prob = -1e100
        # Keep track of number of epochs since last improvement.
        epochs_since_last_improvement = 0
        # Keep track of model with best validation performance.
        best_model_state_dict = None

        epochs = 0
        while True:

            # Train for a single epoch.
            self._neural_posterior.neural_net.train()
            for batch in train_loader:
                optimizer.zero_grad()
                inputs, context = batch[0].to(self._device), batch[1].to(self._device)
                log_prob = self._neural_posterior.log_prob(
                    inputs, context=context, normalize=False
                )
                loss = -torch.mean(log_prob)
                loss.backward()
                clip_grad_norm_(
                    self._neural_posterior.neural_net.parameters(), max_norm=5.0
                )
                optimizer.step()

            epochs += 1

            # Calculate validation performance.
            self._neural_posterior.neural_net.eval()
            log_prob_sum = 0
            with torch.no_grad():
                for batch in val_loader:
                    inputs, context = (
                        batch[0].to(self._device),
                        batch[1].to(self._device),
                    )
                    log_prob = self._neural_posterior.log_prob(
                        inputs, context=context, normalize=False
                    )
                    log_prob_sum += log_prob.sum().item()
            validation_log_prob = log_prob_sum / num_validation_examples

            # Check for improvement in validation performance over previous epochs.
            if validation_log_prob > best_validation_log_prob:
                best_validation_log_prob = validation_log_prob
                epochs_since_last_improvement = 0
                best_model_state_dict = deepcopy(
                    self._neural_posterior.neural_net.state_dict()
                )
            else:
                epochs_since_last_improvement += 1

            # If no validation improvement over many epochs, stop training.
            if epochs_since_last_improvement > stop_after_epochs - 1:
                self._neural_posterior.neural_net.load_state_dict(best_model_state_dict)
                break

        # Update summary.
        self._summary["epochs"].append(epochs)
        self._summary["best-validation-log-probs"].append(best_validation_log_prob)

    @property
    def summary(self):
        return self._summary


class NeuralPotentialFunction:
    """
    Implementation of a potential function for Pyro MCMC which uses a neural density
    estimator to evaluate the likelihood.
    """

    def __init__(self, neural_likelihood, prior, true_observation):
        """
        :param neural_likelihood: Conditional density estimator with 'log_prob' method.
        :param prior: Distribution object with 'log_prob' method.
        :param true_observation: torch.Tensor containing true observation x0.
        """

        self._neural_likelihood = neural_likelihood
        self._prior = prior
        self._true_observation = true_observation

    def __call__(self, inputs_dict):
        """
        Call method allows the object to be used as a function.
        Evaluates the given parameters using a given neural likelihood, prior,
        and true observation.

        :param inputs_dict: dict of parameter values which need evaluation for MCMC.
        :return: torch.Tensor potential ~ -[log q(x0 | theta) + log p(theta)]
        """

        parameters = next(iter(inputs_dict.values()))
        log_likelihood = self._neural_likelihood.log_prob(
            inputs=self._true_observation.reshape(1, -1).to("cpu"),
            context=parameters.reshape(1, -1),
            normalize=False,
        )

        # If prior is uniform we need to sum across last dimension.
        if isinstance(self._prior, distributions.Uniform):
            potential = -(log_likelihood + self._prior.log_prob(parameters).sum(-1))
        else:
            potential = -(log_likelihood + self._prior.log_prob(parameters))

        return potential


class SliceNpNeuralPotentialFunction:
    """
    Implementation of a potential function for Pyro MCMC which uses a classifier
    to evaluate a quantity proportional to the likelihood.
    """

    def __init__(self, posterior, prior, true_observation):
        """
        Args:
            posterior: nn
            prior: torch.distribution, Distribution object with 'log_prob' method.
            true_observation:torch.Tensor containing true observation x0.
        """

        self.prior = prior
        self.posterior = posterior
        self.true_observation = true_observation

    def __call__(self, parameters):
        """
        Call method allows the object to be used as a function.
        Evaluates the given parameters using a given neural likelhood, prior,
        and true observation.

        Args:
            parameters_dict: dict of parameter values which need evaluation for MCMC.

        Returns:
            torch.Tensor potential ~ -[log r(x0, theta) + log p(theta)]

        """

        target_log_prob = (
            self.posterior.log_prob(
                inputs=self.true_observation.reshape(1, -1),
                context=torch.Tensor(parameters).reshape(1, -1),
                normalize=False,
            )
            + self.prior.log_prob(torch.Tensor(parameters)).sum()
        )

        return target_log_prob

    def evaluate(self, point):
        raise NotImplementedError