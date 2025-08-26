from setuptools import find_packages, setup

package_name = 'py_sub'

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
    maintainer='akilasar',  # Or your name
    maintainer_email='akilasar@your_email.com',  # Or your email
    description='A simple ROS 2 Python subscriber node',
    license='Apache License 2.0',  # Or your license
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'subscriber = py_sub.subscriber:main',
        ],
    },
)
