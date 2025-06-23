# HAR-DoReMi: Optimizing Data Mixture for Self-Supervised Human Activity Recognition Across Heterogeneous IMU Datasets

This repository contains the implementation of HAR-DoReMi, a approach for Human Activity Recognition (HAR) that uses the DoReMi (Domain Reweighting with Minimax Optimization) algorithm to handle domain shift in time series data.

## Overview

HAR-DoReMi addresses the challenge of domain shift in human activity recognition by:
- Using a Transformer-based architecture (TRMRec) for time series modeling
- Implementing the DoReMi algorithm for automatic domain reweighting
- Combining MSE and Soft-DTW losses for robust training
- Supporting multiple masking strategies for self-supervised learning

## Environment Setup

### Prerequisites
- Python 3.10.13
- CUDA toolkit (recommended for GPU acceleration)

### 1. Create Virtual Environment

```bash
# Create a virtual environment with Python 3.10.13
python3.10.13 -m venv har_doremi_env

# Activate the virtual environment
# On Linux/macOS:
source har_doremi_env/bin/activate

# On Windows:
# har_doremi_env\Scripts\activate
```

### 2. Clone Repository

```bash
git clone <repository-url>
cd har-doremi
```

### 3. Install Dependencies

```bash
# Upgrade pip to latest version
pip install --upgrade pip

# Install required packages
pip install -r requirements.txt
```

## Quick Start

### Basic Usage

Make sure you have activated the virtual environment, then you can directly run the training script:

```bash
# Activate virtual environment
source har_doremi_env/bin/activate

# Basic training command (using default parameters)
python main.py
```

### Command Line Arguments

#### Data Parameters
- `--domains`: List of domain names for training, options: `HHAR`, `Motion`, `Uci`, `Shoaib`
- `--seq_len`: Sequence length (default: 120)
- `--num_channels`: Number of channels (default: 6)

#### Training Parameters
- `--reference_epochs`: Number of reference model training epochs (default: 200)
- `--num_epochs`: Number of main model training epochs (default: 1000)
- `--batch_size`: Batch size (default: 512)
- `--learning_rate`: Learning rate (default: 0.001)
- `--weight_decay`: Weight decay (default: 0.01)

#### Optimizer Parameters
- `--optimizer_name`: Optimizer type, options: `Adamw`, `Adafactor`
- `--adam_beta1`: Adam optimizer beta1 parameter (default: 0.9)
- `--adam_beta2`: Adam optimizer beta2 parameter (default: 0.98)

#### Learning Rate Scheduler Parameters
- `--lr_scheduler_type`: Learning rate scheduler type, options: `linear`, `cosine`
- `--lr_scheduler_name`: Learning rate scheduler name
- `--num_warmups_ratio`: Warmup steps ratio (default: 0.1)
- `--lr_end`: Final learning rate (default: 1e-4)

#### DoReMi Algorithm Parameters
- `--reweight_eta`: Domain weight update learning rate (default: 0.001)
- `--reweight_eps`: Domain weight smoothing parameter (default: 0.01)
- `--mse_factor`: MSE loss weight (default: 1.0)
- `--dtw_factor`: DTW loss weight (default: 0.01)

#### Masking Parameters
- `--mask_method`: Masking method, options: `spantime_channel`, `time_channel`, `channel`, `spantime`, `time`
- `--time_mask_ratio`: Time step masking ratio (0-100, default: 70)
- `--channel_mask_num`: Number of masked channels (default: 3)

#### Other Parameters
- `--seed`: Random seed (default: 42)
- `--device`: Device selection (default: auto, automatically selects GPU or CPU)
- `--log_name`: Experiment log name (default: HMS)
- `--kernel_size`: Model convolution kernel size (default: 8)


## Data Format and Datasets

### Supported Datasets

This project supports the following human activity recognition datasets by default:

1. **HHAR** - Heterogeneity Human Activity Recognition
2. **Motion** - Motion Sense Dataset
3. **UCI** - UCI Human Activity Recognition
4. **Shoaib** - Shoaib et al. Activity Recognition Dataset

### Data Format Requirements

- **Data Shape**: `(num_samples, sequence_length, num_channels)`
  - `num_samples`: Number of samples
  - `sequence_length`: Time series length (default: 120)
  - `num_channels`: Number of sensor channels (default: 6, including x,y,z axes of accelerometer and gyroscope)

- **Supported File Formats**:
  - `.npy` - NumPy arrays (recommended)

### Data Preprocessing

If you have your own IMU data, you can use the preprocessing scripts in the project:

```bash
# Process 6-DOF IMU data using Mahony filter
python datasets/mahony6.py
```

## Key Features

### 1. TRMRec Model
- Transformer-based time series architecture
- Input embedding with positional encoding
- Domain-aware embedding
- SwiGLU activation function and RMSNorm normalization

### 2. DoReMi Training Algorithm
- Automatic domain weight adjustment
- Reference model training for baseline computation
- Excess loss calculation and domain reweighting
- Support for multiple loss functions (MSE + Soft-DTW)

### 3. Masking Strategies
- **Time Masking**: Mask consecutive time steps
- **Channel Masking**: Mask entire channels
- **Time-Channel Masking**: Combined masking strategy
- **Span Time Masking**: Mask time spans
- **Span Time-Channel Masking**: Combined span masking strategy

### 4. Soft-DTW Loss
- CUDA-accelerated implementation
- Differentiable DTW for time series alignment
- Batch processing support

## Citation

If you use this code in your research, please cite:

```bibtex
@article{ban2025har,
  title={HAR-DoReMi: Optimizing Data Mixture for Self-Supervised Human Activity Recognition Across Heterogeneous IMU Datasets},
  author={Ban, Lulu and Zhu, Tao and Lu, Xiangqing and Qiu, Qi and Han, Wenyong and Li, Shuangjian and Chen, Liming and Wang, Kevin I-Kai and Nie, Mingxing and Wan, Yaping},
  journal={arXiv preprint arXiv:2503.13542},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

For questions or suggestions, please open an issue on GitHub.

---

**Note**: Make sure to properly set up the virtual environment and install all dependencies before running the code. If you encounter any issues, please refer to the troubleshooting section or submit an issue.