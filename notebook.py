# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo",
#     "tinydb==4.8.2",
# ]
# ///

import marimo

__generated_with = "0.14.17"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(
        """
    # Hello Marimo Notebook Template World! 👋📓🌍

    Welcome to the `marimo-notebook-template` repository.  You can find more in the template repository [README](https://github.com/MITLibraries/marimo-notebook-template/blob/main/README.md).
    """
    )
    return


@app.cell
def _(mo):
    import sys

    mo.md(f"""Python version: `{sys.version}`""")
    return


@app.cell
def _(mo):
    mo.md(
        """This notebook exemplifies using [inline dependencies](https://docs.marimo.io/guides/package_management/inlining_dependencies/).  The external package `tinydb` is installed as a dependency via inline dependencies at the top of this `notebook.py` file, then imported and used in the following cell..."""
    )
    return


@app.cell
def _(mo):
    import tempfile

    from tinydb import TinyDB

    with tempfile.TemporaryDirectory() as tmpdir:
        db = TinyDB(f"{tmpdir}/db.json")
        db.insert({"name": "test"})
        results = db.all()

    mo.md(
        f"""
    TinyDB loaded: `OK`<br>
    Results: `{results}`
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        """
    The following cell demonstrates a test embedded into the notebook itself.  As noted in the template README, tests may also be added to the `/tests` folder.
    """
    )
    return


@app.cell
def test_fortytwo_addition():
    assert 42 == 40 + 2
    return


if __name__ == "__main__":
    app.run()
