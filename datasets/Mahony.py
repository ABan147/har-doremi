#!/usr/bin/env python3
"""
Mahony Filter Implementation for IMU Data Processing

This module provides a PoseWorker class that implements the Mahony filter algorithm
for processing IMU (Inertial Measurement Unit) data to estimate orientation and
convert vectors between coordinate frames.
"""

import numpy as np
from scipy.spatial.transform import Rotation
from ahrs.filters import Mahony
from typing import Dict, List, Union, Optional
import warnings

# Suppress scipy warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)

class PoseWorker:
    """
    A worker class for processing IMU data using the Mahony filter algorithm.
    
    This class handles multiple sensor streams simultaneously and maintains
    orientation history for each sensor. It can process 6-axis IMU data
    (accelerometer + gyroscope) to estimate orientation quaternions and
    convert vectors between body frame and NED (North-East-Down) frame.
    
    Attributes:
        mahony_filters (Dict): Dictionary storing Mahony filter instances for each sensor
        orientation_history (Dict): Dictionary storing orientation history for each sensor
        output_data (Dict): Dictionary storing processed output data
        default_quaternion (np.ndarray): Default identity quaternion [w, x, y, z]
    """
    
    def __init__(self, sample_period: float = 0.01, kp: float = 5.0, ki: float = 0.5):
        """
        Initialize the PoseWorker with Mahony filter parameters.
        
        Args:
            sample_period (float): Sampling period in seconds (default: 0.01)
            kp (float): Proportional gain for Mahony filter (default: 5.0)
            ki (float): Integral gain for Mahony filter (default: 0.5)
        """
        self.mahony_filters: Dict[str, Mahony] = {}
        self.orientation_history: Dict[str, Dict] = {}
        self.output_data: Dict[str, Dict] = {}
        self.default_quaternion = np.array([1.0, 0.0, 0.0, 0.0])  # [w, x, y, z]
        
        # Store filter parameters for new sensors
        self.sample_period = sample_period
        self.kp = kp
        self.ki = ki
        
        # Initialize default euler angles
        self.default_euler = self.quaternion_to_euler(self.default_quaternion)
    
    def process_imu_data(self, sensor_data: Dict[str, List[float]]) -> Dict[str, Dict]:
        """
        Process IMU data from multiple sensors.
        
        Args:
            sensor_data (Dict[str, List[float]]): Dictionary mapping sensor names to
                                                 6-element lists [ax, ay, az, gx, gy, gz]
        
        Returns:
            Dict[str, Dict]: Processed data containing quaternions and euler angles
                            for each sensor
        """
        self.output_data.clear()
        
        for sensor_name, imu_data in sensor_data.items():
            try:
                self._process_single_sensor(sensor_name, imu_data)
            except Exception as e:
                print(f"Warning: Error processing sensor '{sensor_name}': {str(e)}")
                continue
        
        return self.output_data.copy()
    
    def _process_single_sensor(self, sensor_name: str, imu_data: List[float]) -> None:
        """
        Process IMU data for a single sensor.
        
        Args:
            sensor_name (str): Name/identifier of the sensor
            imu_data (List[float]): 6-element list [ax, ay, az, gx, gy, gz]
        """
        # Validate input data
        if not self._validate_imu_data(imu_data):
            print(f"Warning: Invalid IMU data for sensor '{sensor_name}', using default quaternion")
            self.output_data[sensor_name] = {
                'quaternion': self.default_quaternion.tolist(),
                'euler_angles': self.default_euler.tolist()
            }
            return
        
        # Extract accelerometer and gyroscope data
        accelerometer = np.array(imu_data[:3])
        gyroscope = np.array(imu_data[3:6])
        
        # Initialize filter and history for new sensors
        if sensor_name not in self.mahony_filters:
            self._initialize_sensor(sensor_name)
        
        # Get previous quaternion
        previous_quaternion = self.orientation_history[sensor_name]['quaternion']
        
        # Update quaternion using Mahony filter
        updated_quaternion = self.mahony_filters[sensor_name].updateIMU(
            previous_quaternion, 
            gyr=gyroscope, 
            acc=accelerometer
        )
        
        # Convert quaternion to euler angles
        euler_angles = self.quaternion_to_euler(updated_quaternion)
        
        # Update history
        self.orientation_history[sensor_name] = {
            'quaternion': updated_quaternion,
            'euler_angles': euler_angles
        }
        
        # Store output data
        self.output_data[sensor_name] = {
            'quaternion': updated_quaternion.tolist(),
            'euler_angles': euler_angles.tolist()
        }
    
    def _validate_imu_data(self, imu_data: List[float]) -> bool:
        """
        Validate IMU data format and content.
        
        Args:
            imu_data (List[float]): IMU data to validate
        
        Returns:
            bool: True if data is valid, False otherwise
        """
        if not isinstance(imu_data, (list, np.ndarray)):
            return False
        
        if len(imu_data) != 6:
            return False
        
        # Check for None values or invalid numbers
        try:
            float_data = [float(x) for x in imu_data]
            return not any(x is None or np.isnan(x) or np.isinf(x) for x in float_data)
        except (ValueError, TypeError):
            return False
    
    def _initialize_sensor(self, sensor_name: str) -> None:
        """
        Initialize Mahony filter and orientation history for a new sensor.
        
        Args:
            sensor_name (str): Name/identifier of the sensor
        """
        self.mahony_filters[sensor_name] = Mahony(
            sample_period=self.sample_period,
            kp=self.kp,
            ki=self.ki
        )
        
        self.orientation_history[sensor_name] = {
            'quaternion': self.default_quaternion.copy(),
            'euler_angles': self.default_euler.copy()
        }
    
    @staticmethod
    def quaternion_to_euler(quaternion: np.ndarray) -> np.ndarray:
        """
        Convert quaternion to Euler angles.
        
        Args:
            quaternion (np.ndarray): Quaternion in [w, x, y, z] format
        
        Returns:
            np.ndarray: Euler angles in [roll, pitch, yaw] format (degrees)
        """
        try:
            rotation = Rotation.from_quat(quaternion)
            return rotation.as_euler('xyz', degrees=True)
        except Exception as e:
            print(f"Warning: Error converting quaternion to euler: {str(e)}")
            return np.array([0.0, 0.0, 0.0])
    
    def transform_to_ned(self, vector: Union[List[float], np.ndarray], 
                        quaternion: Union[List[float], np.ndarray]) -> np.ndarray:
        """
        Transform a vector from body frame to NED (North-East-Down) frame.
        
        Args:
            vector (Union[List[float], np.ndarray]): 3D vector in body frame
            quaternion (Union[List[float], np.ndarray]): Orientation quaternion [w, x, y, z]
        
        Returns:
            np.ndarray: Transformed vector in NED frame
        
        Raises:
            ValueError: If input dimensions are incorrect
        """
        try:
            # Validate inputs
            vector = np.array(vector)
            quaternion = np.array(quaternion)
            
            if vector.shape != (3,):
                raise ValueError(f"Vector must be 3D, got shape {vector.shape}")
            
            if quaternion.shape != (4,):
                raise ValueError(f"Quaternion must be 4D, got shape {quaternion.shape}")
            
            # Convert from [w, x, y, z] to [x, y, z, w] format for scipy
            w, x, y, z = quaternion
            scipy_quaternion = [x, y, z, w]
            
            # Create rotation object and apply transformation
            rotation = Rotation.from_quat(scipy_quaternion)
            transformed_vector = rotation.apply(vector)
            
            return transformed_vector
            
        except Exception as e:
            print(f"Warning: Error in NED transformation: {str(e)}")
            return np.array(vector)  # Return original vector if transformation fails
    
    def get_sensor_orientation(self, sensor_name: str) -> Optional[Dict]:
        """
        Get the current orientation data for a specific sensor.
        
        Args:
            sensor_name (str): Name/identifier of the sensor
        
        Returns:
            Optional[Dict]: Dictionary containing quaternion and euler angles,
                           or None if sensor not found
        """
        if sensor_name in self.orientation_history:
            return self.orientation_history[sensor_name].copy()
        return None
    
    def reset_sensor(self, sensor_name: str) -> None:
        """
        Reset orientation history for a specific sensor.
        
        Args:
            sensor_name (str): Name/identifier of the sensor
        """
        if sensor_name in self.orientation_history:
            self.orientation_history[sensor_name] = {
                'quaternion': self.default_quaternion.copy(),
                'euler_angles': self.default_euler.copy()
            }
    
    def reset_all_sensors(self) -> None:
        """
        Reset orientation history for all sensors.
        """
        for sensor_name in self.orientation_history:
            self.reset_sensor(sensor_name)
    
    def get_active_sensors(self) -> List[str]:
        """
        Get list of currently active sensor names.
        
        Returns:
            List[str]: List of sensor names
        """
        return list(self.mahony_filters.keys())

# Legacy compatibility - maintain old method names
class PoseWorkerLegacy(PoseWorker):
    """
    Legacy compatibility wrapper for the original PoseWorker interface.
    
    This class maintains backward compatibility with the original method names
    while providing the improved functionality of the new implementation.
    """
    
    def process(self, axis6_dict: dict) -> dict:
        """
        Legacy method for processing IMU data (maintains original interface).
        
        Args:
            axis6_dict (dict): Dictionary mapping sensor names to 6-element lists
        
        Returns:
            dict: Processed data with original format
        """
        result = self.process_imu_data(axis6_dict)
        
        # Convert to legacy format
        legacy_result = {}
        for sensor_name, data in result.items():
            legacy_result[sensor_name] = {
                'q': data['quaternion'],
                'e': data['euler_angles']
            }
        
        return legacy_result
    
    def get_q(self, name: str, axis6: list) -> np.ndarray:
        """
        Legacy method for getting quaternion (maintains original interface).
        
        Args:
            name (str): Sensor name
            axis6 (list): 6-element IMU data list
        
        Returns:
            np.ndarray: Quaternion
        """
        self._process_single_sensor(name, axis6)
        if name in self.orientation_history:
            return self.orientation_history[name]['quaternion']
        return self.default_quaternion
    
    @staticmethod
    def q2e(q: np.ndarray) -> np.ndarray:
        """
        Legacy method for quaternion to euler conversion.
        
        Args:
            q (np.ndarray): Quaternion
        
        Returns:
            np.ndarray: Euler angles
        """
        return PoseWorker.quaternion_to_euler(q)
    
    def to_ned(self, a: Union[List[float], np.ndarray], 
              q: Union[List[float], np.ndarray]) -> np.ndarray:
        """
        Legacy method for NED transformation (maintains original interface).
        
        Args:
            a (Union[List[float], np.ndarray]): Vector to transform
            q (Union[List[float], np.ndarray]): Quaternion
        
        Returns:
            np.ndarray: Transformed vector
        """
        return self.transform_to_ned(a, q)

