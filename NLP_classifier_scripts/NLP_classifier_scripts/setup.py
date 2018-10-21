from setuptools import find_packages, setup

REQUIRED_PACKAGES = [
    'tensorflow==1.9.0',
    'h5py',
    'numpy>=1.13.3',
    'tqdm'
]

setup(
    name='plant_classifer',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='Plant classifier application'
)
