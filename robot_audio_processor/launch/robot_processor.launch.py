from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, EnvironmentVariable
from launch.conditions import IfCondition
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Get the package share directory
    pkg_dir = get_package_share_directory('robot_audio_processor')
    
    # Declare launch arguments
    use_kinect = DeclareLaunchArgument(
        'use_kinect',
        default_value='false',
        description='Whether to use the Azure Kinect camera'
    )
    
    # Set up Hugging Face token environment variable
    hf_token = SetEnvironmentVariable(
        'HF_TOKEN',
        EnvironmentVariable('HF_TOKEN', default_value='')
    )
    
    # Include the Azure Kinect launch file only if use_kinect is true
    kinect_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(get_package_share_directory('azure_kinect_ros_driver'),
                        'launch', 'driver.launch.py')
        ]),
        condition=IfCondition(LaunchConfiguration('use_kinect'))
    )
    
    # Launch the audio processor node with HF_TOKEN
    audio_node = Node(
        package='robot_audio_processor',
        executable='audio_processor_node.py',
        name='audio_processor_node',
        output='screen',
        env=[{'name': 'HF_TOKEN', 'value': EnvironmentVariable('HF_TOKEN', default_value='')}]
    )
    
    # Launch the face detection node only if use_kinect is true
    face_node = Node(
        package='robot_audio_processor',
        executable='face_detection_node.py',
        name='face_detection_node',
        output='screen',
        condition=IfCondition(LaunchConfiguration('use_kinect')),
        env=[{'name': 'HF_TOKEN', 'value': EnvironmentVariable('HF_TOKEN', default_value='')}]
    )
    
    # Launch the movement controller node
    movement_node = Node(
        package='robot_audio_processor',
        executable='movement_controller_node.py',
        name='movement_controller_node',
        output='screen'
    )
    
    return LaunchDescription([
        use_kinect,
        hf_token,
        kinect_launch,
        audio_node,
        face_node,
        movement_node
    ]) 