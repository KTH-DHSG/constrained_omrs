from glob import glob
import os

from setuptools import find_packages, setup

package_name = "constrained_omrs"


def files_under(directory: str):
    return [path for path in glob(os.path.join(directory, "**", "*"), recursive=True) if os.path.isfile(path)]


data_files = [
    ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
    (f"share/{package_name}", ["package.xml"]),
    (f"share/{package_name}/launch", glob("launch/*.launch.py")),
    (f"share/{package_name}/config", glob("config/*")),
    (f"share/{package_name}/worlds", glob("worlds/*")),
]

for directory in ("models/omrs_quadrotor",):
    data_files.append(
        (f"share/{package_name}/{directory}", files_under(directory))
    )

setup(
    name=package_name,
    version="0.8.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    data_files=data_files,
    install_requires=["setuptools", "numpy"],
    extras_require={
        "simulation": ["matplotlib", "pillow"],
        "test": ["pytest>=8.0"],
    },
    zip_safe=True,
    maintainer="Constrained OMRS authors",
    maintainer_email="maintainer@example.com",
    description="Constrained open multi-robot formation control with Gazebo validation.",
    license="BSD-3-Clause",
    entry_points={
        "console_scripts": [
            "omrs_gazebo_controller = constrained_omrs_ros.gazebo_controller:main",
        ],
    },
)
