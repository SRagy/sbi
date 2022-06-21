from typing import Any, Callable

from sbi.samplers.importance.importance_sampling import importance_sample
import torch
from torch import Tensor
from tqdm.auto import tqdm


def sampling_importance_resampling(
    potential_fn: Callable,
    proposal: Any,
    num_samples: int = 1,
    num_importance_samples: int = 32,
    max_sampling_batch_size: int = 10_000,
    show_progress_bars: bool = False,
    **kwargs,
) -> Tensor:
    """Perform sampling importance resampling (SIR).

    Args:
        num_samples: Number of samples to draw.
        potential_fn: Potential function, this may be used to debias the proposal.
        proposal: Proposal distribution to propose samples.
        num_importance_samples: Number of proposed samples form which only one is
            selected based on its importance weight.
        num_samples_batch: Number of samples processed in parallel. For large K you may
            want to reduce this, depending on your memory capabilities.

    Returns:
        Tensor: Samples of shape (num_samples, event_shape).
    """

    accepted = []
    sampling_batch_size = min(num_samples, max_sampling_batch_size)

    iters = int(num_samples / sampling_batch_size)

    # Progress bar can be skipped, e.g. when sampling after each round just for logging.
    pbar = tqdm(
        disable=not show_progress_bars,
        total=iters,
        desc=f"Drawing {num_samples} posterior samples",
    )

    for _ in range(iters):
        batch_size = min(sampling_batch_size, num_samples - len(accepted))
        with torch.no_grad():
            thetas, log_weights, _ = importance_sample(
                potential_fn=potential_fn, proposal=proposal, num_samples=batch_size
            )
            weights = log_weights.softmax(-1).cumsum(-1)
            uniform_decision = torch.rand(batch_size, 1, device=thetas.device)
            mask = torch.cumsum(weights >= uniform_decision, -1) == 1
            samples = thetas.reshape(batch_size, num_importance_samples, -1)[mask]
            accepted.append(samples)

        pbar.update(samples.shape[0])
    pbar.close()

    return torch.vstack(accepted)
