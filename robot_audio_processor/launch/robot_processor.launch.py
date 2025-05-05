import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition
from launch_ros.actions import Node

def generate_launch_description():
    # Get the workspace directory
    workspace_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    venv_dir = os.path.join(workspace_dir, 'src', 'venv')
    venv_python = '/home/10_fri/bwi_ros2/src/venv/bin/python3'  # Explicit path to venv Python
    venv_site_packages = os.path.join(venv_dir, 'lib', 'python3.10', 'site-packages')
    
    # Create log directory if it doesn't exist
    log_dir = os.path.join(os.path.expanduser('~'), '.ros', 'log')
    os.makedirs(log_dir, exist_ok=True)
    
    # Get current PYTHONPATH
    current_pythonpath = os.environ.get('PYTHONPATH', '')
    
    # Set up environment variables
    env = {
        'PYTHONPATH': f"{venv_site_packages}:{current_pythonpath}",
        'PATH': f"{os.path.join(venv_dir, 'bin')}:{os.environ.get('PATH', '')}",
        'LD_LIBRARY_PATH': f"{os.path.join(venv_dir, 'lib')}:{os.environ.get('LD_LIBRARY_PATH', '')}",
        'HF_TOKEN': os.environ.get('HF_TOKEN', ''),
        'ROS_LOG_DIR': log_dir,
        'ROS_HOME': os.path.join(os.path.expanduser('~'), '.ros'),
        'ROS_VERSION': '2',
        'ROS_PYTHON_VERSION': '3',
        'ROS_DISTRO': 'humble',
        'PYTHONUNBUFFERED': '1',
        'PYTHONIOENCODING': 'utf-8',
        'VIRTUAL_ENV': venv_dir
    }

    # Declare launch arguments
    use_kinect = LaunchConfiguration('use_kinect', default='false')
    
    # Create nodes with explicit Python interpreter and environment
    audio_node = Node(
        package='robot_audio_processor',
        executable=venv_python,
        arguments=['/home/10_fri/bwi_ros2/install/robot_audio_processor/lib/robot_audio_processor/audio_processor_node.py'],
        name='audio_processor_node',
        output='screen',
        env=env
    )
    
    face_node = Node(
        package='robot_audio_processor',
        executable=venv_python,
        arguments=['/home/10_fri/bwi_ros2/install/robot_audio_processor/lib/robot_audio_processor/face_detection_node.py'],
        name='face_detection_node',
        output='screen',
        env=env
    )
    
    movement_node = Node(
        package='robot_audio_processor',
        executable=venv_python,
        arguments=['/home/10_fri/bwi_ros2/install/robot_audio_processor/lib/robot_audio_processor/movement_controller_node.py'],
        name='movement_controller_node',
        output='screen',
        env=env
    )
    
    # Create launch description
    ld = LaunchDescription()
    
    # Add launch arguments
    ld.add_action(DeclareLaunchArgument(
        'use_kinect',
        default_value='false',
        description='Whether to use Azure Kinect camera'
    ))
    
    # Add nodes
    ld.add_action(audio_node)
    ld.add_action(face_node)
    ld.add_action(movement_node)
    
    return ld 