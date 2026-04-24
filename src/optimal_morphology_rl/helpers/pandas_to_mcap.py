import os
from rclpy.serialization import serialize_message
from std_msgs.msg import Float32
import rosbag2_py

def data_df_to_mcap(df, mcap_folder_path: str):
    """
    Convert a pandas DataFrame to an MCAP file.

    Parameters:
    df (pandas.DataFrame): The input DataFrame to convert.
    mcap_folder_path (str): The folder path to save the MCAP file.
    """

    #remove the folder with mcap_folder_path if it exists
    if os.path.exists(mcap_folder_path):
        import shutil
        shutil.rmtree(mcap_folder_path)

    writer = rosbag2_py.SequentialWriter()
    writer.open(
        rosbag2_py.StorageOptions(uri=mcap_folder_path, storage_id="mcap"),
        rosbag2_py.ConverterOptions(
            input_serialization_format="cdr", output_serialization_format="cdr"
        ),
    )

    for topic in df.columns:
        writer.create_topic(
            rosbag2_py.TopicMetadata(
                name=f"/{topic}", type="std_msgs/msg/Float32", serialization_format="cdr"
            )
        )

    for topic in df.columns:
        for i in range(len(df)):
            msg = Float32()
            msg.data = df.iloc[i][topic]
            timestamp = df.index[i].value
            writer.write(f"/{topic}", serialize_message(msg), timestamp)