# cdps-dashboard

## Developing 
As this dashboard relies on S3 Inventory data, authenticate with `Dev1` credentials before editing.

The recommended approach for developing a Marimo notebook is to use the Marimo GUI editor:

```shell
make edit-notebook
```

### Testing
To run tests:

```shell
make test
```

### Linting
To run linting:

```shell
make lint
```

## Running
Often, notebooks are [served as an "app"](https://docs.marimo.io/guides/apps/).  This is the default mode for [marimo-launcher](https://github.com/MITLibraries/marimo-launcher).

```shell
uv run marimo run --sandbox --headless --no-token notebook.py
```

## Environment Variables

### Required
```shell
S3_INVENTORY_LOCATIONS=# A comma-delimited list of S3 URIs containing S3 Inventory symlink.txt files.
DIGITIZED_BAG_IDS=# A list of bag IDs used by `is_digitized_aip` function. TO BE DEPRECATED IN FUTURE COMMITS
```

### Optional
```shell
# add optional env vars here...
```