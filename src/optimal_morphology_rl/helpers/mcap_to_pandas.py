import pandas as pd
from mcap_ros2.reader import read_ros2_messages

def process_field(topic, field):
    """Process a ROS message field and return a dictionary with meaningful names"""
    if field._type == 'sensor_msgs/Imu':
        return {
            # Orientation
            f"{topic}_orientation_x": field.orientation.x,
            f"{topic}_orientation_y": field.orientation.y,
            f"{topic}_orientation_z": field.orientation.z,
            f"{topic}_orientation_w": field.orientation.w,
            
            # Angular velocity
            f"{topic}_angular_velocity_x": field.angular_velocity.x,
            f"{topic}_angular_velocity_y": field.angular_velocity.y,
            f"{topic}_angular_velocity_z": field.angular_velocity.z,
            
            # Linear acceleration
            f"{topic}_linear_acceleration_x": field.linear_acceleration.x,
            f"{topic}_linear_acceleration_y": field.linear_acceleration.y,
            f"{topic}_linear_acceleration_z": field.linear_acceleration.z,
        }
    if field._type == 'std_msgs/Float32':
        return {
            # Orientation
            f"{topic}_data": field.data,
        }
    return None

def read_mcap_to_dataframe(file_path: str, topics: list = None, range: list = None) -> pd.DataFrame:
    """Read MCAP file to pandas DataFrame with maximum performance."""
    if topics is None:
        topics = [
            "/imu/data_raw",
            "/weight_kg", 
            "/desired_position_rad",
            "/measured_position_rad",
            "/measured_velocity_rad_per_sec",
        ]
    
    msgs = read_ros2_messages(file_path, topics=topics)
    
    # Use list comprehensions for better performance
    processed_data = [
        (msg.log_time_ns, process_field(msg.channel.topic[1:].replace("/", "_"), msg.ros_msg))
        for msg in msgs
    ]
    
    # Filter out None values and separate timestamps from data
    timestamps = []
    data_dicts = []
    
    for timestamp, data in processed_data:
        if data is not None:
            timestamps.append(timestamp)
            data_dicts.append(data)
    
    # Create DataFrame with timestamp as index in one operation
    df = pd.DataFrame(
        data_dicts,
        index=pd.to_datetime(timestamps, unit="ns"),
    ).sort_index()
    
    return df[range[0]:range[1]] if range is not None else df
