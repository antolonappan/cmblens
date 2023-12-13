from distutils.core import setup

setup(
    name="cmblens",
    packages=["cmblens", "cmblens_mini"],
    version="1.2",
    description="Lensing CMB with websky kappa",
    author="Anto Idicherian Lonappan",
    author_email="mail@antolonappan.me",
    url="https://github.com/antolonappan/cmblens",
    install_requires=[
        "numpy",
        "healpy",
        "matplotlib",
        "lenspyx",
        "sqlalchemy",
        "mpi4py",
        "pre-commit",
        "ruff",
        "requests",
    ],
)
