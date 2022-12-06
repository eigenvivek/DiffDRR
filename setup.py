from setuptools import find_packages, setup


# Get the version number from the version.py file
with open("diffdrr/version.py") as f:
    __version__ = f.read().split()[-1].strip("'")

setup(
    name="diffdrr",
    version=__version__,
    packages=find_packages(),
    description="Auto-differentiable digitally reconstructed radiographs in PyTorch.",
    author="Vivek Gopalakrishnan",
    license="MIT",
    install_requires=[
        "torch",
        "pydicom",
        "matplotlib",
        "seaborn",
        "tqdm",
    ],
)
