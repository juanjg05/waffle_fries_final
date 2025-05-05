from setuptools import setup
import os
from glob import glob

package_name = 'robot_audio_processor'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=[
        'setuptools',
        'numpy',
        'scikit-learn',
        'torch',
        'mediapipe',
        'sentencepiece',
        'pytorch_lightning',
        'Cython'
    ],
    zip_safe=True,
    maintainer='juanjg05',
    maintainer_email='juanjgarcia05@gmail.com',
    description='Robot audio processing package',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'audio_processor_node = robot_audio_processor.scripts.audio_processor_node:main',
            'face_detection_node = robot_audio_processor.scripts.face_detection_node:main',
            'movement_controller_node = robot_audio_processor.scripts.movement_controller_node:main',
            'analyze_spoken_to = robot_audio_processor.scripts.analyze_spoken_to:main',
            'analyze_speaker_contexts = robot_audio_processor.scripts.analyze_speaker_contexts:main',
            'export_contexts = robot_audio_processor.scripts.export_contexts:main',
        ],
    },
) 