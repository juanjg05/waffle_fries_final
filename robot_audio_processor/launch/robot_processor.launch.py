from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Get the package share directory
    pkg_dir = get_package_share_directory('robot_audio_processor')
    
    # Include the Azure Kinect launch file
    kinect_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(get_package_share_directory('azure_kinect_ros_driver'),
                        'launch', 'driver.launch.py')
        ])
    )
    
    # Launch the audio processor node
    audio_node = Node(
        package='robot_audio_processor',
        executable='audio_processor_node.py',
        name='audio_processor_node',
        output='screen'
    )
    
    # Launch the face detection node
    face_node = Node(
        package='robot_audio_processor',
        executable='face_detection_node.py',
        name='face_detection_node',
        output='screen'
    )
    
    return LaunchDescription([
        kinect_launch,
        audio_node,
        face_node
    ]) 