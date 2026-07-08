from setuptools import find_packages, setup

package_name = 'spot_mpc_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    package_data={
        package_name: ['plans/*.json'],
    },
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='akilasar',
    maintainer_email='akilasar@todo.todo',
    description='Sampling-MPC controller and debug visualizer for real Spot hallway navigation',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'spot_mpc_node = spot_mpc_pkg.spot_mpc_node:main',
            'spot_mpc_visualizer = spot_mpc_pkg.spot_mpc_visualizer:main',
        ],
    },
)
