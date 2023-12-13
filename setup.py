from distutils.core import setup

files = ["cmblens/*"]
setup(
    name="cmblens",
    packages=["cmblens"],
    package_data={"cmblens": files},
    version="1.1",
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
    ],
)
