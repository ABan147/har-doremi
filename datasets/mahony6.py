#!/usr/bin/env python3
"""
Mahony Algorithm Data Processing Script

This script processes IMU (Inertial Measurement Unit) data using the Mahony filter algorithm
to convert accelerometer and gyroscope data from body frame to NED (North-East-Down) frame.
The processed data is then saved for further analysis.

"""

from pathlib import Path
import numpy as np
import os
from tqdm import tqdm
from Mahony import PoseWorker
from ahrs.filters import Mahony
from matplotlib import pyplot as plt
import warnings

# Suppress matplotlib warnings for cleaner output
warnings.filterwarnings('ignore')

def process_imu_data_with_mahony(data_array):
    """
    Process IMU data using Mahony filter to convert from body frame to NED frame.
    
    Args:
        data_array (np.ndarray): Input IMU data with shape (n_samples, 6)
                                 where columns 0-2 are accelerometer data (ax, ay, az)
                                 and columns 3-5 are gyroscope data (gx, gy, gz)
    
    Returns:
        np.ndarray: Processed data in NED frame with same shape as input
    """
    # Initialize Mahony filter and pose worker
    pose_worker = PoseWorker()
    orientation_filter = Mahony()
    
    num_samples = len(data_array)
    
    # Pre-allocate arrays for efficiency
    processed_data = np.empty((num_samples - 1, 6))
    quaternions = np.tile([1., 0., 0., 0.], (num_samples, 1))  # Initialize quaternions
    
    # Process each sample using Mahony filter
    print(f"Processing {num_samples} samples with Mahony filter...")
    for t in tqdm(range(1, num_samples), desc="Mahony filtering"):
        # Update quaternion using IMU data
        quaternions[t] = orientation_filter.updateIMU(
            quaternions[t - 1], 
            gyr=data_array[t, 3:6],  # Gyroscope data
            acc=data_array[t, 0:3]   # Accelerometer data
        )
        
        # Transform accelerometer and gyroscope data to NED frame
        acc_ned = pose_worker.transform_to_ned(data_array[t, 0:3], quaternions[t])
        gyr_ned = pose_worker.transform_to_ned(data_array[t, 3:6], quaternions[t])
        
        # Concatenate transformed data
        processed_data[t - 1] = np.concatenate((acc_ned, gyr_ned), axis=0)
    
    # Insert the first sample (unchanged) at the beginning
    final_data = np.insert(processed_data, 0, data_array[0], axis=0)
    
    return final_data

def save_visualization(data, output_path, title_suffix=""):
    """
    Save visualization of the Z-axis data.
    
    Args:
        data (np.ndarray): Data to visualize
        output_path (str): Path to save the plot
        title_suffix (str): Suffix for the plot title
    """
    plt.figure(figsize=(10, 6))
    plt.plot(data[:, 2], linewidth=1.5)
    plt.title(f'Z-axis Data {title_suffix}')
    plt.xlabel('Sample Index')
    plt.ylabel('Z-axis Value')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def main():
    """
    Main function to process all datasets in the specified folder.
    """
    # Define input folder path
    folder_path = Path('dataset/dataset_merge_origin')
    # folder_path = Path('har-doremi/datasets/orginal/shoaib/')
    
    if not folder_path.exists():
        print(f"Error: Folder {folder_path} does not exist!")
        return
    
    print(f"Processing datasets in: {folder_path}")
    print("=" * 50)
    
    # Check if data_20_120.npy exists directly in the folder
    dataset_path = folder_path / 'data_20_120.npy'
    
    if dataset_path.exists():
        # Process the dataset directly in the current folder
        dataset_name = folder_path.name  # Use folder name as dataset name
        print(f"\nProcessing dataset: {dataset_name}")
        print(f"Dataset path: {dataset_path}")
        
        try:
            # Load and reshape data
            raw_data = np.load(dataset_path)
            print(f"Raw data shape: {raw_data.shape}")
            
            # Reshape to (n_samples, 6) format
            data_array = np.reshape(raw_data, (-1, 6))
            print(f"Reshaped data: {data_array.shape}")
            print(f"First sample: {data_array[0]}")
            
            # Save original data visualization
            original_plot_path = folder_path / f'{dataset_name}_original.png'
            save_visualization(data_array, original_plot_path, "(Original)")
            print(f"Saved original data visualization to: {original_plot_path}")
            
            # Optional: Convert acceleration from g to m/s² (uncomment if needed)
            # data_array[:, :3] *= 9.80665  # Convert g to m/s²
            
            # Process data using Mahony filter
            processed_data = process_imu_data_with_mahony(data_array)
            print(f"Processed data shape: {processed_data.shape}")
            
            # Save processed data visualization
            processed_plot_path = folder_path / f'{dataset_name}_NED.png'
            save_visualization(processed_data, processed_plot_path, "(NED Frame)")
            print(f"Saved processed data visualization to: {processed_plot_path}")
            
            # Reshape to final format (assuming 120 samples per window)
            final_data = np.reshape(processed_data, (-1, 120, 6))
            print(f"Final data shape: {final_data.shape}")
            
            # Save processed data
            output_path = folder_path / f'{dataset_name}_AG_NED.npy'
            np.save(output_path, final_data)
            print(f"Saved processed data to: {output_path}")
            
        except Exception as e:
            print(f"Error processing {dataset_name}: {str(e)}")
            return
    else:
        # Iterate through all subdirectories in the dataset folder (original logic)
        found_datasets = False
        for item in folder_path.iterdir():
            if not item.is_dir():
                continue
                
            found_datasets = True
            print(f"\nProcessing dataset: {item.name}")
            print(f"Dataset path: {item}")
            
            # Load the dataset
            sub_dataset_path = item / 'data_20_120.npy'
            
            if not sub_dataset_path.exists():
                print(f"Warning: {sub_dataset_path} not found, skipping...")
                continue
                
            try:
                # Load and reshape data
                raw_data = np.load(sub_dataset_path)
                print(f"Raw data shape: {raw_data.shape}")
                
                # Reshape to (n_samples, 6) format
                data_array = np.reshape(raw_data, (-1, 6))
                print(f"Reshaped data: {data_array.shape}")
                print(f"First sample: {data_array[0]}")
                
                # Save original data visualization
                original_plot_path = item / f'{item.name}_original.png'
                save_visualization(data_array, original_plot_path, "(Original)")
                print(f"Saved original data visualization to: {original_plot_path}")
                
                # Optional: Convert acceleration from g to m/s² (uncomment if needed)
                # data_array[:, :3] *= 9.80665  # Convert g to m/s²
                
                # Process data using Mahony filter
                processed_data = process_imu_data_with_mahony(data_array)
                print(f"Processed data shape: {processed_data.shape}")
                
                # Save processed data visualization
                processed_plot_path = item / f'{item.name}_NED.png'
                save_visualization(processed_data, processed_plot_path, "(NED Frame)")
                print(f"Saved processed data visualization to: {processed_plot_path}")
                
                # Reshape to final format (assuming 120 samples per window)
                final_data = np.reshape(processed_data, (-1, 120, 6))
                print(f"Final data shape: {final_data.shape}")
                
                # Save processed data
                output_path = item / f'{item.name}_AG_NED.npy'
                np.save(output_path, final_data)
                print(f"Saved processed data to: {output_path}")
                
            except Exception as e:
                print(f"Error processing {item.name}: {str(e)}")
                continue
        
        if not found_datasets:
            print(f"Warning: No subdirectories found and no data_20_120.npy in {folder_path}")
    
    print("\n" + "=" * 50)
    print("Data processing completed!")

if __name__ == "__main__":
    main()

