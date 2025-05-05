from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
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
    
    # Include the Azure Kinect launch file only if use_kinect is true
    kinect_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(get_package_share_directory('azure_kinect_ros_driver'),
                        'launch', 'driver.launch.py')
        ]),
        condition=IfCondition(LaunchConfiguration('use_kinect'))
    )
    
    # Get HF_TOKEN from environment
    hf_token = os.environ.get('HF_TOKEN', '')
    
    # Get the virtual environment path
    venv_path = os.path.join(os.path.expanduser('~'), 'bwi_ros2', 'src', 'venv')
    python_executable = os.path.join(venv_path, 'bin', 'python3')
    
    # Get the current PYTHONPATH
    current_pythonpath = os.environ.get('PYTHONPATH', '')
    
    # Set up environment variables
    env_vars = {
        'HF_TOKEN': hf_token,
        # Prioritize virtual environment packages
        'PYTHONPATH': f"{venv_path}/lib/python3.10/site-packages:{current_pythonpath}",
        'PATH': f"{venv_path}/bin:{os.environ.get('PATH', '')}",
        'LD_LIBRARY_PATH': f"/opt/ros/humble/lib:{os.environ.get('LD_LIBRARY_PATH', '')}",
        'ROS_VERSION': '2',
        'ROS_PYTHON_VERSION': '3',
        'ROS_DISTRO': 'humble',
        # Add these to ensure proper Python package loading
        'PYTHONUNBUFFERED': '1',
        'PYTHONIOENCODING': 'utf-8'
    }
    
    # Launch the audio processor node with environment variables
    audio_node = Node(
        package='robot_audio_processor',
        executable=python_executable,
        arguments=['/home/10_fri/bwi_ros2/install/robot_audio_processor/lib/robot_audio_processor/audio_processor_node.py'],
        name='audio_processor_node',
        output='screen',
        env=env_vars
    )
    
    # Launch the face detection node only if use_kinect is true
    face_node = Node(
        package='robot_audio_processor',
        executable=python_executable,
        arguments=['/home/10_fri/bwi_ros2/install/robot_audio_processor/lib/robot_audio_processor/face_detection_node.py'],
        name='face_detection_node',
        output='screen',
        condition=IfCondition(LaunchConfiguration('use_kinect')),
        env=env_vars
    )
    
    # Launch the movement controller node
    movement_node = Node(
        package='robot_audio_processor',
        executable=python_executable,
        arguments=['/home/10_fri/bwi_ros2/install/robot_audio_processor/lib/robot_audio_processor/movement_controller_node.py'],
        name='movement_controller_node',
        output='screen',
        env=env_vars
    )
    
    return LaunchDescription([
        use_kinect,
        kinect_launch,
        audio_node,
        face_node,
        movement_node
    ]) 