import math
import warnings
import numpy as np
import torch
from torch.utils.data import Sampler
from torch.optim.lr_scheduler import LRScheduler
from transformers import get_scheduler
from transformers.utils import ExplicitEnum

def custom_collate_fn(batch):
    """
    Custom collate function for DataLoader.
    """
    inputs = torch.stack([item['input'] for item in batch])
    domain_ids = torch.tensor([item['domain_id'] for item in batch])
    
    return {
        'input': inputs,
        'domain_id': domain_ids
    }


class DomainWeightedSampler(Sampler):
    """
    Custom sampler that samples from different domains based on given weights.
    Uses GPU acceleration and more precise sample allocation.
    """
    def __init__(self, datasets, domain_weights, num_samples):
        self.datasets = datasets
        self.domain_weights = torch.tensor(domain_weights, device='cuda')  # Use GPU
        self.num_samples = num_samples

        self.dataset_sizes = [len(d) for d in datasets]
        self.cumulative_sizes = torch.cumsum(torch.tensor([0] + self.dataset_sizes), dim=0).to('cuda')

        # Calculate floating-point values for samples per domain
        self.samples_per_domain_float = self.domain_weights * num_samples
        self.samples_per_domain = torch.floor(self.samples_per_domain_float).long()

        # Handle cases where total samples might be insufficient
        total_samples = torch.sum(self.samples_per_domain)
        missing_samples = num_samples - total_samples.item()

        # Calculate decimal parts
        decimal_parts = self.samples_per_domain_float - self.samples_per_domain.float()

        # Use numpy lexsort for sorting by decimal parts and weights
        # Need to move tensors back to CPU for numpy operations
        weighted_indices = np.lexsort((-self.domain_weights.cpu().numpy(), -decimal_parts.cpu().numpy()))

        # Fill missing samples from domains with larger decimal parts
        for idx in weighted_indices:
            while missing_samples > 0 and decimal_parts[idx] > 0:
                self.samples_per_domain[idx] += 1
                missing_samples -= 1
                decimal_parts[idx] = 0  # Prevent refilling

    def __iter__(self):
        indices = []
        for domain, samples_number in enumerate(self.samples_per_domain):
            if samples_number <= self.dataset_sizes[domain]:
                # Sample without replacement
                sampled_indices_in_domain = np.random.choice(
                    self.dataset_sizes[domain],
                    size=samples_number.item(),
                    replace=False
                )
            else:
                # Sample all data and then sample additional data with replacement
                sampled_indices_in_domain = np.arange(self.dataset_sizes[domain])
                additional_samples = samples_number - self.dataset_sizes[domain]
                additional_indices = np.random.choice(
                    self.dataset_sizes[domain],
                    size=additional_samples.item(),
                    replace=True
                )
                sampled_indices_in_domain = np.concatenate([sampled_indices_in_domain, additional_indices])

            # Convert to global indices
            global_indices = self.cumulative_sizes[domain] + torch.tensor(sampled_indices_in_domain).to('cuda')
            indices.extend(global_indices.tolist())

        # Shuffle the indices to mix domains
        np.random.shuffle(indices)

        return iter(indices)

    def __len__(self):
        return self.num_samples


class LinearWarmupExponentialLR(LRScheduler):
    """
    Learning rate scheduler with linear warmup and exponential decay to a specified end learning rate.
    """
    def __init__(self, optimizer, num_warmup_steps, num_training_steps, lr_start=1e-7, lr_end=0, last_epoch=-1, verbose=False):
        # Initialize parameters
        self.num_warmup_steps = num_warmup_steps  # Number of warmup steps
        self.num_training_steps = num_training_steps  # Total training steps
        self.lr_start = lr_start  # Initial learning rate
        self.lr_end = lr_end  # End learning rate
        super().__init__(optimizer, last_epoch, verbose)  # Call parent constructor

    def get_lr(self):
        # Get current learning rate
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, please use `get_last_lr()`.", UserWarning)

        if self.last_epoch > self.num_training_steps:
            # If current step exceeds total training steps, return current learning rate
            return [group['lr'] for group in self.optimizer.param_groups]

        return self._get_closed_form_lr()  # Return closed-form learning rate

    def _get_closed_form_lr(self):
        # Calculate closed-form learning rate
        if self.last_epoch < self.num_warmup_steps:
            # During warmup phase, linearly increase learning rate
            return [self.lr_start + (base_lr - self.lr_start) * self.last_epoch / self.num_warmup_steps for base_lr in self.base_lrs]
        else:
            # Calculate decay rate to make learning rate approach lr_end at training end
            gammas = [np.exp(np.log(1e-10 / (base_lr - self.lr_end)) / (self.num_training_steps - self.num_warmup_steps))
                      for base_lr in self.base_lrs]
            # Return decayed learning rate
            return [self.lr_end + (base_lr - self.lr_end) * gamma ** (self.last_epoch - self.num_warmup_steps) for base_lr, gamma in zip(self.base_lrs, gammas)]


class LinearWarmupCosineLR(LRScheduler):
    """
    Cosine learning rate scheduler with linear warmup and decay to a specified end learning rate.
    """
    def __init__(self, optimizer, num_warmup_steps, num_training_steps, lr_start=1e-7, lr_end=0, last_epoch=-1, verbose=False):
        # Initialize scheduler parameters
        self.num_warmup_steps = num_warmup_steps  # Number of warmup steps
        self.num_training_steps = num_training_steps  # Total training steps
        self.lr_start = lr_start  # Initial learning rate
        self.lr_end = lr_end  # End learning rate
        super().__init__(optimizer, last_epoch, verbose)  # Call parent constructor

    def get_lr(self):
        # Get current learning rate
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, please use `get_last_lr()`.", UserWarning)

        if self.last_epoch > self.num_training_steps:
            # If current step exceeds total training steps, return current learning rate
            return [group['lr'] for group in self.optimizer.param_groups]

        return self._get_closed_form_lr()  # Return closed-form learning rate

    def _get_closed_form_lr(self):
        # Calculate closed-form learning rate
        if self.last_epoch < self.num_warmup_steps:
            # During warmup phase, linearly increase learning rate
            return [self.lr_start + (base_lr - self.lr_start) * self.last_epoch / self.num_warmup_steps for base_lr in self.base_lrs]
        else:
            # Calculate cosine decayed learning rate
            return [self.lr_end + (base_lr - self.lr_end) * (1 + math.cos(math.pi * (self.last_epoch - self.num_warmup_steps) / (self.num_training_steps - self.num_warmup_steps))) / 2 for base_lr in self.base_lrs]


class ExtendedSchedulerType(ExplicitEnum):
    # Define extended scheduler type enumeration
    LINEAR_WARMUP_EXPONENTIAL = "linear_warmup_exponential"  # Linear warmup exponential scheduler
    LINEAR_WARMUP_COSINE = "linear_warmup_cosine"  # Linear warmup cosine scheduler


# Extended scheduler function mapping
TYPE_TO_EXTENDED_SCHEDULER_FUNCTION = {
    ExtendedSchedulerType.LINEAR_WARMUP_EXPONENTIAL: LinearWarmupExponentialLR,  # Map to linear warmup exponential scheduler
    ExtendedSchedulerType.LINEAR_WARMUP_COSINE: LinearWarmupCosineLR  # Map to linear warmup cosine scheduler
}


def get_scheduler_extended(
    name,
    optimizer,
    num_warmup_steps=0,
    num_training_steps=0,
    lr_end=1e-4,
):
    """
    Get extended scheduler.

    Args:
        name (str): Name of the scheduler.
        optimizer: Optimizer to be scheduled.
        num_warmup_steps (int): Number of warmup steps.
        num_training_steps (int): Total training steps.
        lr_end (float): End learning rate.

    Returns:
        Scheduler instance.
    """
    try:
        # Try to convert name to extended scheduler type
        name = ExtendedSchedulerType(name)
        schedule_func = TYPE_TO_EXTENDED_SCHEDULER_FUNCTION[name]  # Get corresponding scheduler function
    except ValueError:
        # If name is invalid, return default scheduler
        return get_scheduler(name, optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)

    # All other schedulers require `num_training_steps`
    if num_training_steps is None:
        raise ValueError(f"{name} requires `num_training_steps`, please provide that argument.")

    # Return scheduler instance
    return schedule_func(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps, lr_end=lr_end)


def span_mask(seq_len, max_gram=3, p=0.2, goal_num_predict=15):
    ngrams = np.arange(1, max_gram + 1, dtype=np.int64)
    pvals = p * np.power(1 - p, np.arange(max_gram))
    pvals /= pvals.sum(keepdims=True)
    mask_pos = set()
    while len(mask_pos) < goal_num_predict:
        n = np.random.choice(ngrams, p=pvals)
        n = min(n, goal_num_predict - len(mask_pos))
        anchor = np.random.randint(seq_len)
        if anchor in mask_pos:
            continue
        for i in range(anchor, min(anchor + n, seq_len - 1)):
            mask_pos.add(i)
    return list(mask_pos)


def mask_input(x, method='time', time_mask_ratio=70, channel_mask_num=None):
    """
    Masks the input tensor x according to the specified type.

    Args:
        x: Input tensor of shape (batch_size, seq_len, n_features)
        method: Type of masking to apply
        time_mask_ratio: Percentage of time steps to mask (0-100)
        channel_mask_num: Number of channels to mask (if applicable)

    Returns:
        x_masked: Masked input tensor
        mask: Mask tensor indicating masked positions (1 for unmasked, 0 for masked)
    """
    batch_size, seq_len, n_features = x.size()

    x_masked = x.clone().detach()

    # Initialize masks
    mask_time = torch.ones((batch_size, seq_len), device=x.device)
    mask_channel = torch.ones((batch_size, n_features), device=x.device)

    num_masked_time = max(int(seq_len * time_mask_ratio * 0.01), 1)

    if method in ['time', 'time_channel']:
        # Vectorized per-sample time masks
        rand = torch.rand(batch_size, seq_len, device=x.device)
        _, time_indices = torch.topk(rand, k=num_masked_time, dim=1, largest=False)
        batch_indices = torch.arange(batch_size).unsqueeze(1).expand(batch_size, num_masked_time).to(x.device)
        mask_time[batch_indices, time_indices] = 0
    elif method in ['spantime', 'spantime_channel']:
        # For 'spantime' methods, loop over batch_size
        for i in range(batch_size):
            time_index = span_mask(seq_len, goal_num_predict=num_masked_time)
            mask_time[i, time_index] = 0

    if method in ['spantime_channel', 'time_channel', 'channel']:
        if channel_mask_num is None:
            channel_mask_num = max(int(n_features * 0.3), 1)  # Default to masking 30% of channels
        # Vectorized per-sample channel masks
        rand_channel = torch.rand(batch_size, n_features, device=x.device)
        _, channel_indices = torch.topk(rand_channel, k=channel_mask_num, dim=1, largest=False)
        batch_indices = torch.arange(batch_size).unsqueeze(1).expand(batch_size, channel_mask_num).to(x.device)
        mask_channel[batch_indices, channel_indices] = 0

    # Expand masks to match the dimensions of x
    mask_time = mask_time.unsqueeze(-1)  # Shape: (batch_size, seq_len, 1)
    mask_channel = mask_channel.unsqueeze(1)  # Shape: (batch_size, 1, n_features)

    mask = mask_time * mask_channel  # Broadcasting to shape (batch_size, seq_len, n_features)

    x_masked = x_masked * mask

    return x_masked, mask