## CMB Lensing for LiteBIRD simulation

### Installation

`pip install .`

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
