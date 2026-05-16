from glob import glob
import os

from setuptools import find_packages, setup

package_name = 'nl_planner'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    package_data={
        package_name: [
            'prompts/*.md',
        ],
    },
    include_package_data=True,
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=[
        'setuptools',
        'pydantic>=2.5',
        'pydantic-ai-slim[openai]>=0.0.10',
        'pyyaml>=6.0',
        'openai>=1.40',
    ],
    zip_safe=True,
    maintainer='akilasar',
    maintainer_email='akilasar@mit.edu',
    description=(
        'English natural-language mission to STL + cluster-plan pipeline.'
    ),
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'nl_planner       = nl_planner.cli:main',
            'seed_taxonomy    = nl_planner.bootstrap.seed_taxonomy:main',
            'planner_node     = nl_planner.nodes.planner_node:main',
            'executor_node    = nl_planner.nodes.executor_node:main',
        ],
    },
)
