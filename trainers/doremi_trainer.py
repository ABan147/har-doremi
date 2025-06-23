import copy
import torch
import torch.nn as nn
import traceback
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim import AdamW
from torch.cuda.amp import GradScaler
from transformers import Adafactor
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils.soft_dtw_cuda import SoftDTW
from utils.training_utils import custom_collate_fn, DomainWeightedSampler, get_scheduler_extended, mask_input

class DoReMiTrainer:
    """
    DoReMi (Domain Reweighting with Minimax Optimization) Trainer for HAR tasks.
    
    This trainer implements the DoReMi algorithm for domain adaptation in human activity recognition,
    using dynamic domain reweighting based on excess loss computation.
    """
    
    def __init__(self, model, reference_model, datasets, args):
        self.model = model
        self.reference_model = reference_model
        self.datasets = datasets
        self.args = args
        self.create_optimizer()
        self.lr_scheduler = None

        self.batch_counter = 0

        self.avg_domain_weights = torch.zeros(len(datasets))
        # Initialize per-domain scores and update counter
        self.perdomain_scores = torch.zeros(len(datasets))
        self.update_counter = 0

        # Split datasets into training and validation sets
        self.ref_datasets = []
        for dataset in datasets:
            self.ref_datasets.append(dataset)

        # Create a ConcatDataset for training and validation
        self.ref_concat_dataset = ConcatDataset(self.ref_datasets)

        self.domain_weights = [max(1.0 / len(datasets), 1e-8) for _ in datasets]
        ref_total_size = sum(len(dataset) for dataset in self.ref_datasets)
        self.initial_domain_weights = [len(dataset) / ref_total_size for dataset in self.ref_datasets]
        self.reference_num_samples = ref_total_size
        self.train_num_samples = self.args.batch_size

        # Set domain weights for the reference model
        self.reference_domain_weights = copy.deepcopy(self.initial_domain_weights)

        self.domain_sampling_weights = copy.deepcopy(self.domain_weights)

        # Update dataloader with initial domain weights
        self.update_dataloader()
        self.create_reference_dataloader()

        num_training_steps = args.num_epochs * len(self.train_dataloader)
        self.lr_scheduler = self.create_scheduler(num_training_steps)
        self.num_training_steps = num_training_steps

        self.scaler = GradScaler()

        # For visualization
        self.weight_history = {i: [] for i in range(len(datasets))}

        # Initialize accumulators for domain_ids
        self.accumulated_domain_ids = []

        # Initialize accumulators for excess_loss
        self.accumulated_excess_loss = []

        self.log_name = self.args.log_name

        self.writer = SummaryWriter(log_dir=f"logs/{self.log_name}")

    def create_scheduler(self, num_training_steps, optimizer=None):
        if optimizer is None:
            optimizer = self.optimizer
        if self.args.lr_scheduler_name is not None:
            lr_scheduler_name = self.args.lr_scheduler_name
        else:
            lr_scheduler_name = self.args.lr_scheduler_type
        scheduler = get_scheduler_extended(
            lr_scheduler_name,
            optimizer=optimizer,
            num_warmup_steps=int(self.args.num_warmups_ratio * num_training_steps),
            num_training_steps=num_training_steps,
            lr_end=self.args.lr_end,
        )
        return scheduler

    def create_optimizer(self):
        no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        if self.args.optimizer_name.lower() == "adafactor":
            self.optimizer = Adafactor(
                optimizer_grouped_parameters,
                lr=self.args.learning_rate,
                scale_parameter=False,
                relative_step=False,
                beta1=self.args.adam_beta1,
                warmup_init=False,
                weight_decay=self.args.weight_decay,
            )
            # Set beta1 for Adafactor optimizer
            self.optimizer.beta1 = self.args.adam_beta1
            self.optimizer.beta2 = self.args.adam_beta2
            for param_group in self.optimizer.param_groups:
                param_group['beta1'] = self.args.adam_beta1
                param_group['beta2'] = self.args.adam_beta2
        else:
            self.optimizer = AdamW(optimizer_grouped_parameters,
                                   lr=self.args.learning_rate,
                                   weight_decay=self.args.weight_decay,
                                   betas=(self.args.adam_beta1, self.args.adam_beta2)
                                   )
        return self.optimizer



    def update_dataloader(self):
        # Ensure weights are positive and normalized
        total_weight = sum(self.domain_weights)
        if total_weight == 0:
            print("Total domain weight is zero, resetting domain weights")
            self.reset_domain_weights()
            total_weight = sum(self.domain_weights)
        # Add epsilon to domain weights to prevent zeros
        self.domain_weights = [max(w / total_weight, 1e-8) for w in self.domain_weights]
        weight_sum = sum(self.domain_weights)
        self.domain_weights = [w / weight_sum for w in self.domain_weights]

        # Update sampling weights
        self.domain_sampling_weights = copy.deepcopy(self.domain_weights)

        # Create the custom sampler
        num_samples = self.train_num_samples
        sampler = DomainWeightedSampler(
            datasets=self.ref_datasets,
            domain_weights=self.domain_sampling_weights,
            num_samples=num_samples
        )

        self.train_dataloader = DataLoader(
            self.ref_concat_dataset,
            batch_size=self.args.batch_size,
            sampler=sampler,
            num_workers=0,
            collate_fn=custom_collate_fn
        )

    def update_domain_weights(self):
        # Concatenate accumulated excess losses and domain IDs
        all_excess_loss = torch.cat(self.accumulated_excess_loss)
        all_domain_ids = torch.cat(self.accumulated_domain_ids)

        domain_scores = []
        for d in range(len(self.datasets)):
            domain_mask = (all_domain_ids == d)
            if domain_mask.sum() > 0:
                domain_excess_loss = all_excess_loss[domain_mask]
                # min=0 means only focus on the part where proxy_model loss is greater than reference_model loss in each data domain
                domain_excess_loss = torch.clip(domain_excess_loss, min=0)
                positive_excess_loss = domain_excess_loss[domain_excess_loss > 0]
                if len(positive_excess_loss) > 0:
                    domain_score = positive_excess_loss.mean().item()
                else:
                    domain_score = 0.0
            else:
                domain_score = self.perdomain_scores[d].item()
            domain_scores.append(domain_score)

        # Update per-domain scores
        self.perdomain_scores = torch.tensor(domain_scores).float()
        print(f"\nperdomain_scores before: {self.perdomain_scores}")

        # Handle numerical stability
        self.perdomain_scores = torch.nan_to_num(self.perdomain_scores, nan=0.0, posinf=1e6, neginf=-1e6)

        # Read current domain weights
        train_domain_weights = torch.tensor(self.domain_weights).float()
        train_domain_weights = torch.clamp(train_domain_weights, min=1e-8)

        # Update log weights
        eta = self.args.reweight_eta
        log_new_domain_weights = torch.log(train_domain_weights) + eta * self.perdomain_scores
        log_new_domain_weights = log_new_domain_weights - torch.logsumexp(log_new_domain_weights, dim=0)
        new_domain_weights = torch.exp(log_new_domain_weights)

        # Apply epsilon smoothing
        eps = self.args.reweight_eps
        num_domains = len(self.datasets)
        new_domain_weights = (1 - eps) * new_domain_weights + eps / num_domains

        # Ensure domain weights sum to 1 and are above a minimum threshold
        min_weight = 1e-8
        new_domain_weights = torch.clamp(new_domain_weights, min=min_weight)
        new_domain_weights = new_domain_weights / new_domain_weights.sum()

        # Update domain weights
        self.domain_weights = new_domain_weights.tolist()

        print("Domain weights: ", self.domain_weights)

        # Update dataloader with new weights
        self.update_dataloader()

        # Record weights for visualization
        for i, w in enumerate(self.domain_weights):
            self.weight_history[i].append(w)

        # Update running average of domain weights
        self.avg_domain_weights = [
            (self.avg_domain_weights[i] * (self.update_counter - 1) + self.domain_weights[i]) / self.update_counter
            for i in range(len(self.domain_weights))
        ]

    def compute_softdtw_loss(self, true_seq, pred_seq, dim=1):
        """
        Compute similarity loss between true sequence (true_seq) and reconstructed sequence (pred_seq) using Soft-DTW.
        true_seq, pred_seq shape: (batch_size, seq_len, n_channels)
        """
        batch_size, seq_len, n_channels = true_seq.shape

        # Initialize Soft-DTW with CUDA enabled
        softdtw = SoftDTW(use_cuda=self.args.device.type == 'cuda', gamma=0.1)  # gamma controls softening degree, adjustable

        # Keep original shape (batch_size, seq_len, n_channels), directly input to SoftDTW
        true_data = true_seq.detach()  # Remove gradient tracking
        pred_data = pred_seq.detach()  # Remove gradient tracking

        # Compute Soft-DTW loss
        loss = softdtw(true_data, pred_data)  # Shape: (batch_size,)
        dtw_losses = loss / n_channels  # Average Soft-DTW distance across all channels

        dtw_losses = dtw_losses

        return dtw_losses.to(self.args.device, dtype=torch.float32)

    def train_reference_model(self, num_epochs=100):
        # Create optimizer and scheduler for the reference model
        self.create_reference_optimizer()
        num_training_steps = num_epochs * len(self.reference_dataloader)
        self.reference_scheduler = self.create_scheduler(num_training_steps, optimizer=self.reference_optimizer)

        # Initialize variables to record losses (use dictionary to store MSE and DTW losses)
        best_domain_losses = {'mse': [float('inf')] * len(self.datasets), 'dtw': [float('inf')] * len(self.datasets)}
        total_domain_losses = {'mse': [0] * len(self.datasets), 'dtw': [0] * len(self.datasets)}

        for epoch in range(num_epochs):
            self.reference_model.train()
            epoch_domain_losses = {'mse': [0] * len(self.datasets), 'dtw': [0] * len(self.datasets)}
            epoch_domain_counts = [0] * len(self.datasets)

            for batch in self.reference_dataloader:
                inputs = batch['input'].to(self.args.device)
                domain_ids = batch['domain_id'].to(self.args.device)

                try:
                    masked_inputs, mask = mask_input(inputs, method=self.args.mask_method,
                                                          time_mask_ratio=self.args.time_mask_ratio,
                                                          channel_mask_num=self.args.channel_mask_num)
                    outputs = self.reference_model(masked_inputs, domain_ids, mask)

                    # 1. Compute MSE loss (masked reconstruction loss)
                    # Compute loss only at masked positions
                    loss_fn = nn.MSELoss(reduction='none')
                    mse_loss = loss_fn(outputs, inputs)  # Shape: (batch_size, seq_len, n_channels)
                    loss_mask = (1 - mask)  # Shape: (batch_size, seq_len, n_channels)
                    # Apply loss mask
                    masked_mse_loss = mse_loss * loss_mask  # Shape: (batch_size, seq_len, n_channels)
                    # Sum over time steps and channels to get per-sample loss
                    per_sample_mse_loss = masked_mse_loss.sum(dim=[1, 2]) / loss_mask.sum(dim=[1, 2]).clamp(
                        min=1e-8)  # Shape: (batch_size,)

                    # 2. Compute DTW loss (sequence trend loss)
                    per_sample_dtw_loss = self.compute_softdtw_loss(inputs, outputs)

                    # Compute curr_domain_weights inversely proportional to domain_weights and sampling probabilities
                    train_domain_weights = torch.tensor(self.reference_domain_weights).to(inputs.device).float()
                    train_domain_weights = torch.clamp(train_domain_weights, min=1e-8)

                    # Compute sampling ratios
                    sampling_weights = []
                    total_data_points = len(domain_ids)

                    for d in range(len(self.datasets)):
                        domain_mask = (domain_ids == d)
                        count = domain_mask.sum()
                        if count > 0:
                            ratio = float(count / total_data_points)
                            sampling_weights.append(ratio)
                        else:
                            sampling_weights.append(0)

                    sampling_weights = torch.tensor(sampling_weights).to(inputs.device).float()
                    sampling_weights = torch.clamp(sampling_weights, min=1e-8)

                    # Adjust for sampling weights
                    adjusted_domain_weights = train_domain_weights / sampling_weights
                    adjusted_domain_weights = adjusted_domain_weights / adjusted_domain_weights.sum()

                    curr_domain_weights = adjusted_domain_weights[domain_ids].detach()

                    # Compute total loss (combine MSE and DTW losses)
                    total_loss = (self.args.mse_factor * per_sample_mse_loss + self.args.dtw_factor * per_sample_dtw_loss) * curr_domain_weights
                    total_loss = total_loss.sum()

                    # Normalize total_loss
                    normalizer = curr_domain_weights.sum()
                    normalizer = torch.clamp(normalizer, min=1e-10)
                    total_loss = total_loss / normalizer

                    # Backpropagation
                    scaled_loss = self.scaler.scale(total_loss)
                    scaled_loss.backward()

                    # Gradient clipping
                    self.scaler.unscale_(self.reference_optimizer)
                    torch.nn.utils.clip_grad_norm_(self.reference_model.parameters(), max_norm=1.0)

                    self.scaler.step(self.reference_optimizer)
                    self.scaler.update()
                    self.reference_optimizer.zero_grad()
                    self.reference_scheduler.step()

                    # Record per-domain losses for both MSE and DTW
                    for d in range(len(self.datasets)):
                        domain_mask = (domain_ids == d)
                        if domain_mask.sum() > 0:
                            epoch_domain_counts[d] += domain_mask.sum().float()
                            # MSE loss
                            epoch_domain_losses['mse'][d] += per_sample_mse_loss[domain_mask].sum().float()
                            # DTW loss
                            epoch_domain_losses['dtw'][d] += per_sample_dtw_loss[domain_mask].sum().float()

                except Exception as e:
                    print(f"Exception during reference model training: {e}")
                    traceback.print_exc()
                    raise e

            # Update total losses and best losses for both MSE and DTW
            for d in range(len(self.datasets)):
                if epoch_domain_counts[d] > 0:
                    # MSE
                    avg_mse_loss = epoch_domain_losses['mse'][d] / epoch_domain_counts[d]
                    total_domain_losses['mse'][d] += avg_mse_loss
                    if avg_mse_loss < best_domain_losses['mse'][d]:
                        best_domain_losses['mse'][d] = avg_mse_loss
                    # DTW
                    avg_dtw_loss = epoch_domain_losses['dtw'][d] / epoch_domain_counts[d]
                    total_domain_losses['dtw'][d] += avg_dtw_loss
                    if avg_dtw_loss < best_domain_losses['dtw'][d]:
                        best_domain_losses['dtw'][d] = avg_dtw_loss

            print(f"Reference Model Epoch {epoch + 1}/{num_epochs} completed.")

        # Compute average domain losses over all epochs for both MSE and DTW
        average_domain_losses = {
            'mse': [total / num_epochs for total in total_domain_losses['mse']],
            'dtw': [total / num_epochs for total in total_domain_losses['dtw']]
        }
        self.reference_average_domain_losses = average_domain_losses
        self.reference_best_domain_losses = best_domain_losses

        # Compute baseline losses
        self.compute_baseline_loss()

        # Delete reference model to free up memory
        del self.reference_model

    def create_reference_dataloader(self):
        num_samples = self.reference_num_samples
        sampler = DomainWeightedSampler(
            datasets=self.ref_datasets,
            domain_weights=self.reference_domain_weights,
            num_samples=num_samples
        )

        self.reference_dataloader = DataLoader(
            self.ref_concat_dataset,
            batch_size=self.args.batch_size,
            sampler=sampler,
            num_workers=0,
            collate_fn=custom_collate_fn
        )

    def create_reference_optimizer(self):
        no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.reference_model.named_parameters() if
                           not any(nd in n for nd in no_decay)],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in self.reference_model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        if self.args.optimizer_name.lower() == "adafactor":
            self.reference_optimizer = Adafactor(
                optimizer_grouped_parameters,
                lr=self.args.learning_rate,
                scale_parameter=False,
                relative_step=False,
                beta1=self.args.adam_beta1,
                warmup_init=False,
                weight_decay=self.args.weight_decay,
            )
        else:
            self.reference_optimizer = AdamW(
                optimizer_grouped_parameters,
                lr=self.args.learning_rate,
                weight_decay=self.args.weight_decay,
                betas=(self.args.adam_beta1, self.args.adam_beta2)
            )

    def compute_baseline_loss(self):
        self.baseline_losses = {
            'mse': [],
            'dtw': []
        }
        for avg_loss, best_loss in zip(self.reference_average_domain_losses['mse'],
                                       self.reference_best_domain_losses['mse']):
            baseline_mse_loss = 0.7 * avg_loss + 0.3 * best_loss
            self.baseline_losses['mse'].append(baseline_mse_loss)
        for avg_loss, best_loss in zip(self.reference_average_domain_losses['dtw'],
                                       self.reference_best_domain_losses['dtw']):
            baseline_dtw_loss = 0.7 * avg_loss + 0.3 * best_loss
            self.baseline_losses['dtw'].append(baseline_dtw_loss)
        print("Baseline losses per domain (MSE):", self.baseline_losses['mse'])
        print("Baseline losses per domain (DTW):", self.baseline_losses['dtw'])



    def train_step(self, batch):
        if len(batch['input']) == 0:
            return 0, {'mse': [0] * len(self.datasets), 'dtw': [0] * len(self.datasets)}, [0] * len(self.datasets)
        
        self.model.train()
        inputs = batch['input'].to(self.args.device)
        domain_ids = batch['domain_id'].to(self.args.device)

        try:
            masked_inputs, mask = mask_input(inputs, method=self.args.mask_method,
                                                  time_mask_ratio=self.args.time_mask_ratio,
                                                  channel_mask_num=self.args.channel_mask_num)
            outputs = self.model(masked_inputs, domain_ids, mask)

            # 1. Compute MSE loss (masked reconstruction loss)
            # Compute loss only at masked positions
            loss_fn = nn.MSELoss(reduction='none')
            mse_loss = loss_fn(outputs, inputs)  # Shape: (batch_size, seq_len, n_channels)
            loss_mask = (1 - mask)  # Shape: (batch_size, seq_len, n_channels)
            # Apply loss mask
            masked_mse_loss = mse_loss * loss_mask  # Shape: (batch_size, seq_len, n_channels)
            # Sum over time steps and channels to get per-sample loss
            per_sample_mse_loss = masked_mse_loss.sum(dim=[1, 2]) / loss_mask.sum(dim=[1, 2]).clamp(
                min=1e-8)  # Shape: (batch_size,)

            # 2. Compute DTW loss (sequence trend loss)
            per_sample_dtw_loss = self.compute_softdtw_loss(inputs, outputs)

            # print("per_sample_mse_loss", per_sample_mse_loss)
            # print("per_sample_dtw_loss", per_sample_dtw_loss)

            # Get baseline losses per domain (get from dictionary)
            baseline_losses = {
                'mse': torch.tensor(self.baseline_losses['mse']).to(inputs.device),
                'dtw': torch.tensor(self.baseline_losses['dtw']).to(inputs.device)
            }
            baseline_mse_loss_per_sample = baseline_losses['mse'][domain_ids]  # Shape: (batch_size,)
            baseline_dtw_loss_per_sample = baseline_losses['dtw'][domain_ids]  # Shape: (batch_size,)

            # Compute excess loss for both MSE and DTW
            excess_mse_loss = per_sample_mse_loss - baseline_mse_loss_per_sample
            excess_dtw_loss = per_sample_dtw_loss - baseline_dtw_loss_per_sample

            # Combined excess loss (weighted combination of MSE and DTW excess losses)
            excess_loss = self.args.mse_factor * excess_mse_loss + self.args.dtw_factor * excess_dtw_loss  # w1, w2 are weights

            # Accumulate excess losses and domain IDs
            self.accumulated_excess_loss.append(excess_loss.detach())
            self.accumulated_domain_ids.append(domain_ids.detach())

            # Compute curr_domain_weights inversely proportional to domain_weights and sampling probabilities
            train_domain_weights = torch.tensor(self.domain_weights).to(inputs.device).float()
            train_domain_weights = torch.clamp(train_domain_weights, min=1e-8)

            # Compute adjusted domain weights
            sampling_weights = []
            total_data_points = len(domain_ids)

            for d in range(len(self.datasets)):
                domain_mask = (domain_ids == d)
                count = domain_mask.sum()  # Number of samples from domain d in this batch
                if count > 0:
                    ratio = float(count / total_data_points)  # Sampling ratio
                    sampling_weights.append(ratio)
                else:
                    sampling_weights.append(0)

            sampling_weights = torch.tensor(sampling_weights).to(inputs.device).float()
            sampling_weights = torch.clamp(sampling_weights, min=1e-8)

            # Adjust for sampling weights
            adjusted_domain_weights = train_domain_weights / sampling_weights
            adjusted_domain_weights = adjusted_domain_weights / adjusted_domain_weights.sum()

            curr_domain_weights = adjusted_domain_weights[domain_ids].detach()

            # Compute total loss (combine MSE and DTW losses)
            total_loss = (self.args.mse_factor * per_sample_mse_loss + self.args.dtw_factor * per_sample_dtw_loss) * curr_domain_weights
            total_loss = total_loss.sum()

            # Normalize total_loss
            normalizer = curr_domain_weights.sum()
            normalizer = torch.clamp(normalizer, min=1e-10)
            total_loss = total_loss / normalizer

            # Backpropagation
            scaled_loss = self.scaler.scale(total_loss)
            scaled_loss.backward()

            # Gradient clipping
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
            self.lr_scheduler.step()

            # Update domain weights
            self.update_domain_weights()

            # Increment update_counter here
            self.update_counter += 1  # Count update times

            # Reset accumulators
            self.accumulated_excess_loss = []
            self.accumulated_domain_ids = []

            # Increment step_counter after backward pass
            self.batch_counter += 1

            # Record per-domain losses (return dictionary format)
            domain_losses = {'mse': torch.zeros(len(self.datasets), device=self.args.device),
                             'dtw': torch.zeros(len(self.datasets), device=self.args.device)}
            domain_samples = torch.zeros(len(self.datasets), device=self.args.device)

            for d in range(len(self.datasets)):
                domain_mask = (domain_ids == d)
                if domain_mask.sum() > 0:
                    domain_samples[d] += domain_mask.sum()
                    # MSE loss
                    domain_losses['mse'][d] += per_sample_mse_loss[domain_mask].sum()
                    # DTW loss
                    domain_losses['dtw'][d] += per_sample_dtw_loss[domain_mask].sum()

            # Convert tensors to Python floats for return and visualization
            domain_losses_list = {'mse': domain_losses['mse'].tolist(), 'dtw': domain_losses['dtw'].tolist()}
            domain_samples_list = domain_samples.tolist()

            return total_loss.item(), domain_losses_list, domain_samples_list

        except Exception as e:
            print(f"Exception during training step at batch {self.batch_counter}: {e}")
            traceback.print_exc()
            raise e

    def train(self):
        for epoch in range(self.args.num_epochs):
            self.batch_counter = 0  # Reset batch counter at the start of each epoch
            self.model.train()

            device = self.args.device

            epoch_losses = {'mse': torch.zeros(len(self.datasets), device=device),
                            'dtw': torch.zeros(len(self.datasets), device=device)}
            epoch_counts = torch.zeros(len(self.datasets), device=device)

            for step, batch in enumerate(tqdm(self.train_dataloader, desc=f"Epoch {epoch}")):
                loss, domain_losses, domain_samples = self.train_step(batch)

                # Convert 'mse' and 'dtw' losses in dictionary to tensors respectively
                domain_losses_mse_tensor = torch.tensor(domain_losses['mse'], device=device)
                domain_losses_dtw_tensor = torch.tensor(domain_losses['dtw'], device=device)
                domain_samples_tensor = torch.tensor(domain_samples, device=device)

                # Accumulate losses and sample counts for each domain
                epoch_losses['mse'] += domain_losses_mse_tensor
                epoch_losses['dtw'] += domain_losses_dtw_tensor
                epoch_counts += domain_samples_tensor

            # Compute average losses over accumulation interval
            total_loss_mse = epoch_losses['mse'].sum()
            total_loss_dtw = epoch_losses['dtw'].sum()
            total_samples = epoch_counts.sum()
            avg_epoch_train_loss = (
                        (total_loss_mse + total_loss_dtw) / total_samples).item() if total_samples > 0 else 0.0

            # Compute per-domain average losses
            # Prevent division by zero
            domain_average_losses = {
                'mse': [loss / count if count > 0 else 0.0 for loss, count in
                        zip(epoch_losses['mse'].tolist(), epoch_counts.tolist())],
                'dtw': [loss / count if count > 0 else 0.0 for loss, count in
                        zip(epoch_losses['dtw'].tolist(), epoch_counts.tolist())]
            }

        averages_domain_weights = [sum(values) / len(values) for key, values in sorted(self.weight_history.items())]
        print("averages_domain_weights:", averages_domain_weights)