## CMB Lensing for LiteBIRD simulation

### Installation

`pip install .`

**NOTE:** When installing on NERSC, build `mpi4py` as mentioned [here](https://docs.nersc.gov/development/languages/python/parallel-python/#mpi4py-in-your-custom-conda-environment) before installing the package.

After cloning for the first time, run `pre-commit install`.

### Usage

- `cmblens` module:

    ```python
    import cmblens

    cmblens.CMBLensed(...)
    cmblens.MetaSIM(...)
    cmblens.hash_maps(...)
    cmblens.camb_clfile(...)
    ```

- `cmblens_mini` module:

    ```python
    import cmblens_mini
    
    cmblens_mini.CMBLensed(...)
    cmblens_mini.MetaSIM(...)
    cmblens_mini.hash_maps(...)
    cmblens_mini.camb_clfile(...)
    ```
