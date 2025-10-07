# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo",
# ]
# ///

import marimo

__generated_with = "0.16.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md("""# CDPS Dashboard""")
    return


@app.cell
def _(mo):
    file_count = {"total": "3"}
    file_extensions = {"pdf": "1", "tiff": "2"}
    storage = {"total": "1234"}

    data = mo.ui.dropdown(
        options={
            "File - Count": file_count,
            "File - Extensions": file_extensions,
            "Storage": storage,
        },
        label="Select a data type:",
    )
    data
    return (data,)


@app.cell
def _(data, mo):
    import json

    if not data.selected_key:
        markdown_str = ""
    else:
        markdown_str = f"""
            ## {data.selected_key}
            {json.dumps(data.value)}
            """
    mo.md(markdown_str)
    return


if __name__ == "__main__":
    app.run()
