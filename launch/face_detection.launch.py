from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # Launch Azure Kinect camera
        Node(
            package='azure_kinect_ros_driver',
            executable='node',
            name='azure_kinect_ros_bridge',
            output='screen',
            parameters=[{
                'depth_enabled': True,
                'depth_mode': 'NFOV_UNBINNED',
                'color_enabled': True,
                'color_format': 'bgra',
                'color_resolution': '720P',
                'fps': 30,
                'point_cloud': True,
                'rgb_point_cloud': True,
                'point_cloud_in_depth_frame': True,
                'required': True,
                'imu_rate_target': 0,
                'audio_enabled': True,
                'audio_sample_rate': 48000,
                'audio_channels': 7,
            }]
        ),
        
        # Launch our face detection node
        Node(
            package='robot_audio_processor',
            executable='spoken_to_model.py',
            name='spoken_to_model',
            output='screen'
        )
    ]) 