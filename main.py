#!/usr/bin/env python3
"""
HAR-DoReMi: Human Activity Recognition with Domain Reweighting using Multi-domain Invariant Learning

Main training script for the HAR-DoReMi framework.
This script implements the DoReMi algorithm for domain adaptation in human activity recognition.
"""

import numpy as np
import random
import torch
import argparse
from torch.utils.data import Dataset
from transformers.trainer_utils import set_seed
from models.trm_rec_model import TRMRec
from trainers.doremi_trainer import DoReMiTrainer


def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


class TimeSeriesDataset(Dataset):
    """Dataset class for time series data with domain information."""
    
    def __init__(self, data, domain_id):
        self.data = data  # data shape: (num_sequences, sequence_length, n_channels)
        self.domain_id = domain_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sequence = self.data[idx]
        return {
            'input': torch.tensor(sequence, dtype=torch.float32),
            'domain_id': torch.tensor(self.domain_id, dtype=torch.long),
        }


def load_data(dataset_name, domain_id, sequence_length):
    """Load dataset from predefined paths."""
    data_paths = {
        'HHAR': 'har-doremi-main/datasets/mahony/hhar/data_20_120.npy',
        'Shoaib': 'har-doremi-main/datasets/mahony/shoaib/data_20_120.npy',
        'Motion': 'har-doremi-main/datasets/mahony/motion/data_20_120.npy',
        'Uci': 'har-doremi-main/datasets/mahony/uci/data_20_120.npy',
    }

    if dataset_name not in data_paths:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Use numpy.memmap to memory-map the file
    data = np.load(data_paths[dataset_name], mmap_mode='r')

    # Handle 3D data
    if data.ndim == 3:
        # Check if sequence length matches
        if data.shape[1] != sequence_length:
            # Truncate or pad if sequence length doesn't match
            if data.shape[1] > sequence_length:
                data = data[:, :sequence_length, :]
            else:
                padding = np.zeros((data.shape[0], sequence_length - data.shape[1], data.shape[2]))
                data = np.concatenate((data, padding), axis=1)
        # Data shape is now (num_sequences, sequence_length, n_channels)

    elif data.ndim == 2:
        # If data is 2D, reshape to 3D
        batch_inside_size = sequence_length
        max_n = data.shape[0] // batch_inside_size
        data = data[:batch_inside_size * max_n]
        data = data.reshape(-1, sequence_length, data.shape[1])  # (num_sequences, sequence_length, n_channels)
    else:
        raise ValueError(f"Data has unsupported number of dimensions: {data.ndim}")

    return TimeSeriesDataset(data, domain_id)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train HAR-DoReMi model")
    
    # Data arguments
    parser.add_argument("--domains", type=str, nargs="+", 
                       default=["HHAR","Motion", "Uci"],
                       help="List of domain names to use for training",
                       choices=['HHAR', 'Motion', 'Uci', 'Shoaib'])
    parser.add_argument("--seq_len", type=int, default=120,
                       help="Sequence length")
    parser.add_argument("--num_channels", type=int, default=6,
                        help="Number of channels")
    
    # Training arguments
    parser.add_argument("--reference_epochs", type=int, default=200,
                       help="Number of reference model training epochs")
    parser.add_argument("--num_epochs", type=int, default=1000,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=512,
                       help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=0.001,
                       help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                       help="Weight decay")
    
    # Learning rate scheduler parameters
    parser.add_argument("--lr_scheduler_type", type=str, default="linear", 
                       help="Learning rate scheduler type",choices=['linear', 'cosine'])
    parser.add_argument("--lr_scheduler_name", type=str, default="linear_warmup_cosine",
                       help="Learning rate scheduler name",choices=['linear_warmup_cosine','linear_warmup_exponential'])
    parser.add_argument("--num_warmups_ratio", type=float, default=0.1,
                       help="Number of warmup steps as a ratio of total training steps")
    parser.add_argument("--lr_end", type=float, default=1e-4,
                       help="The final learning rate of the learning rate scheduler")
    
    # Optimizer parameters
    parser.add_argument("--optimizer_name", type=str, default="AdamW",
                       help="Optimizer name",choices=['Adamw','Adafactor'])
    parser.add_argument("--adam_beta1", type=float, default=0.9,
                       help="Adam beta1 parameter")
    parser.add_argument("--adam_beta2", type=float, default=0.98,
                       help="Adam beta2 parameter")
    
    # DoReMi arguments
    parser.add_argument("--reweight_eta", type=float, default=0.001,
                       help="Learning rate for domain weight updates")
    parser.add_argument("--reweight_eps", type=float, default=0.01,
                       help="Epsilon smoothing for domain weights")
    parser.add_argument("--mse_factor", type=float, default=1.0,
                       help="Weight for MSE loss")
    parser.add_argument("--dtw_factor", type=float, default=0.01,
                       help="Weight for DTW loss")
    
    # Masking parameters
    parser.add_argument("--mask_method", type=str, default="spantime_channel",
                       help="Method for masking",choices=['spantime_channel', 'time_channel', 'channel','spantime', 'time'])
    parser.add_argument("--time_mask_ratio", type=int, default=70,
                       help="Ratio of time steps to mask (0-100)")
    parser.add_argument("--channel_mask_num", type=int, default=3,
                       help="Number of channels to mask")
    
    # Model parameters
    parser.add_argument("--kernel_size", type=int, default=8,
                       help="Kernel size for the model")
    
    # Other arguments
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use (auto, cpu, cuda)")
    parser.add_argument("--log_name", type=str, default="HMS",
                       help="Experiment name for logging")
    
    return parser.parse_args()


def setup_config(args):
    """Setup configuration from arguments."""
    # Setup device
    if args.device == "auto":
        args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        args.device = torch.device(args.device)
    
    return args


def main():
    """Main training function."""
    args = setup_config(parse_args())
    set_seed(args.seed)

    # Load and filter datasets
    datasets = [load_data(name, i, args.seq_len) for i, name in enumerate(args.domains)]
    datasets = [ds for ds in datasets if ds is not None]

    print("Dataset sizes:", [f"Domain {i}: {len(ds)}" for i, ds in enumerate(datasets)])

    # Create models with shared parameters
    model_params = dict(n_channels=args.num_channels, n_steps=args.seq_len, 
                       kernel_size=args.kernel_size, num_domains=len(datasets))
    model, reference_model = [TRMRec(**model_params).to(args.device) for _ in range(2)]

    # Initialize trainer and start training
    trainer = DoReMiTrainer(model, reference_model, datasets, args)
    print(f"Parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Domain weights - Proxy: {trainer.domain_weights}, Reference: {trainer.reference_domain_weights}")
    
    trainer.train_reference_model(num_epochs=args.reference_epochs)
    trainer.train()


if __name__ == "__main__":
    main()
