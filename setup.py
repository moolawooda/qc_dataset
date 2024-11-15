from setuptools import find_packages, setup  # type: ignore

setup(
    name="qcds",
    version="0.1.0",
    description="A dataset management tool for quantum computing",
    packages=find_packages(),
    package_data={"qcds": ["templates/*"]},
    include_package_data=True,
)
