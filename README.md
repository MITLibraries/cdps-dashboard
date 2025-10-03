# marimo-notebook-template

**_NOTE: Rework this section to describe your Marimo notebook(s)._**

This repository is a template for a single, or multiple, Marimo notebooks that will be served and accessed via [marimo-launcher](https://github.com/MITLibraries/marimo-launcher).

By default, [marimo-launcher](https://github.com/MITLibraries/marimo-launcher) expects a single file called `notebook.py` at the root of the repository it clones and launches.  However, `marimo-launcher` also supports a `--path` CLI arg or `NOTEBOOK_PATH` env var that allows overriding this default behavior, allowing a Marimo notebook repository like this to have multiple notebooks and/or a unique structure.  This template exemplifies a default structure of a single [notebook.py](notebook.py) file at the root of the repository.

## Developing 

The recommended approach for developing a Marimo notebook is to use the Marimo GUI editor:

```shell
make edit-notebook
```

### Dependencies

There are many ways in which [dependencies can be managed](https://docs.marimo.io/guides/package_management/) for a Marimo notebook.  For details on [marimo-launcher](https://github.com/MITLibraries/marimo-launcher) handles and expects dependencies, please [click here](https://github.com/MITLibraries/marimo-launcher?tab=readme-ov-file#notebook-dependencies).

### Testing

Testing is performed on the command line against inlined tests in notebook(s) ([read more here in the Marimo docs](https://docs.marimo.io/guides/testing/pytest/#testing-at-the-command-line)) and any tests discovered in `/tests`.

There are two primary ways to add tests:

1. Create a cell in a notebook and either manually in the `.py` file, or via the cell action menu, **name** the cell to start with a `test_` prefix.
2. Add tests to the standalone `/tests` folder.  This can be helpful for testing code that may live outside of the notebook itself.

To run tests:

```shell
make test
```

Note the use of `*.py` in the Makefile command `test`.  Per this greedy test discovery approach, _any_ function that starts with `test_` will be included in the test suite.  To avoid this behavior `*.py` can be replaced with more granular, specific filepaths.

### Linting

`mypy` type and `ruff` general linting are similar to other python projects, but note the relaxed rules in `pyproject.toml` for Marimo notebook files.  If multiple notebooks are present, they will need to be added here.

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