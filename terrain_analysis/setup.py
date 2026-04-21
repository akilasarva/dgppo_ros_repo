from setuptools import find_packages, setup

package_name = 'terrain_analysis'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='user@todo.todo',
    description='Lidar Intensity Terrain Classifier',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
    'console_scripts': [
        'classifier = terrain_analysis.terrain_classifier:main',
        'hsv = terrain_analysis.terrain_segmenter:main',
        'segmenter = terrain_analysis.cityscapes_segmenter:main', # New line
        'segformer = terrain_analysis.segformer_node:main',
        'terrain_fusion = terrain_analysis.fusion_node:main',
        ],
},
    )
