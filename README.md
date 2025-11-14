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
# add required env vars here...
```

### Optional
```shell
# add optional env vars here...
```