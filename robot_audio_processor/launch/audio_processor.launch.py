from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='robot_audio_processor',
            executable='audio_processor_node.py',
            name='audio_processor_node',
            output='screen',
            parameters=[{
                'use_sim_time': False,
            }]
        )
    ]) 