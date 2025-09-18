from setuptools import find_packages, setup

package_name = 'dgppo_ros_node_pkg'

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
    maintainer='akilasar',
    maintainer_email='akilasar@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
    'console_scripts': [
        'dgppo_ros_node = dgppo_ros_node_pkg.dgppo_ros_node:main',
        'carla_bridge_node = dgppo_ros_node_pkg.carla_bridge_node:main',
        ],
},
)
