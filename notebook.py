# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo",
#     "pandas==2.3.3",
#     "pyarrow==21.0.0",
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
def _():
    # Functions

    import math
    from pathlib import Path

    import pandas as pd

    def convert_size(size_bytes):
        """Convert byte counts into a human readable format."""
        if size_bytes == 0:
            return "0B"
        size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
        i = math.floor(math.log(size_bytes, 1024))
        p = math.pow(1024, i)
        s = round(size_bytes / p, 2)
        return f"{s} {size_name[i]}"

    def parse_s3_keys(dataframe: pd.DataFrame) -> pd.DataFrame:
        """Parse S3 keys to extract additional metadata and update dataframe."""
        key_parts = dataframe["key"].str.split("/", expand=True)

        dataframe.loc[:, "bagname"] = key_parts[8]
        uuid_parts = [
            key_parts[0] + key_parts[1],
            key_parts[2],
            key_parts[3],
            key_parts[4],
            key_parts[5] + key_parts[6] + key_parts[7],
        ]
        dataframe.loc[:, "uuid"] = "-".join(str(part) for part in uuid_parts)

        dataframe.loc[:, "file"] = dataframe["key"].str.split("/").str[-1]
        dataframe.loc[:, "filepath"] = (
            dataframe["key"].str.split("/").str[9:].apply("/".join)
        )
        dataframe.loc[:, "extension"] = dataframe["filepath"].apply(
            lambda x: Path(x).suffix.lower()
        )
        return dataframe

    def is_metadata(dataframe: pd.DataFrame) -> pd.DataFrame:
        """Identifies metadata files in the DataFrame."""
        metadata_files = [
            "data/logs",
            "data/METS",
            "data/README.html",
            "data/objects/metadata",
            "data/objects/submissionDocumentation",
            "bag-info.txt",
            "bagit.txt",
            "manifest-sha256.txt",
            "tagmanifest-sha256.txt",
        ]
        dataframe.loc[:, "is_metadata"] = dataframe["key"].apply(
            lambda x: any(metadata_file in x for metadata_file in metadata_files)
        )

        return dataframe

    return Path, convert_size, is_metadata, parse_s3_keys, pd


@app.cell
def _(Path, is_metadata, parse_s3_keys, pd):
    # Generate inventory dataframe

    import os

    parquet_files = Path(os.environ["INVENTORY_LOCATIONS"]).glob("*.parquet")
    inventory_df = (
        pd.concat((pd.read_parquet(f) for f in parquet_files), ignore_index=True)
        .drop_duplicates()
        .reset_index(drop=True)
    )

    inventory_df.loc[:, "is_current"] = (
        inventory_df["is_latest"] & ~inventory_df["is_delete_marker"]
    )

    inventory_df = parse_s3_keys(inventory_df)
    inventory_df = is_metadata(inventory_df)
    cdps_df = inventory_df.loc[inventory_df["is_current"]].copy()
    return (cdps_df,)


@app.cell
def _(cdps_df, mo):
    # Files

    file_count = (
        cdps_df.groupby("bucket")
        .size()
        .to_frame("file count")
        .sort_values(by="file count", ascending=False)
    )
    file_extensions = (
        cdps_df.groupby("extension")
        .size()
        .to_frame("file count")
        .sort_values(by="file count", ascending=False)
    )
    file_storage = (
        cdps_df.sort_values(by="size", ascending=False)
        .loc[:, ["file", "size"]]
        .reset_index(drop=True)[:10]
    )
    file_metadata = (
        cdps_df.groupby("is_metadata")
        .size()
        .rename(index={False: "content files", True: "metadata files"})
        .to_frame("file count")
    )

    files_display = mo.vstack(
        [
            mo.md("#### File count by bucket"),
            file_count,
            mo.md("#### File count by extension"),
            file_extensions,
            mo.md("#### Largest 10 files"),
            file_storage,
            mo.md("#### Content vs metadata files"),
            file_metadata,
        ],
        gap=1,
    )
    return (files_display,)


@app.cell
def _(mo):
    # Storage

    storage = {"not implemented": "not implemented"}

    storage_display = mo.vstack(
        [storage],
        gap=1,
    )
    return (storage_display,)


@app.cell
def _(mo):
    # AIPs

    aips = {"not implemented": "not implemented"}

    aip_display = mo.vstack(
        [aips],
        gap=1,
    )
    return (aip_display,)


@app.cell
def _(mo):
    # Digitized vs born-digital content

    digitized_born_digital = {"not implemented": "not implemented"}

    digitized_born_digital_display = mo.vstack(
        [digitized_born_digital],
        gap=1,
    )
    return (digitized_born_digital_display,)


@app.cell
def _(mo):
    # Image vs AV

    image_av = {"not implemented": "not implemented"}

    image_av_display = mo.vstack(
        [image_av],
        gap=1,
    )
    return (image_av_display,)


@app.cell
def _(mo):
    # Original vs duplicate files

    original_duplicate = {"not implemented": "not implemented"}

    original_duplicate_display = mo.vstack(
        [original_duplicate],
        gap=1,
    )
    return (original_duplicate_display,)


@app.cell
def _(cdps_df, convert_size, mo):
    # Summary stats

    total_files = mo.stat(
        label="Total files",
        value=f"{len(cdps_df)}",
    )

    total_storage = mo.stat(
        label="Total storage",
        value=f"{convert_size(cdps_df["size"].sum())}",
    )

    summary = mo.hstack([total_files, total_storage], widths="equal", gap=1)
    return (summary,)


@app.cell
def _(
    aip_display,
    digitized_born_digital_display,
    files_display,
    image_av_display,
    mo,
    original_duplicate_display,
    storage_display,
    summary,
):
    # Dashboard

    accordion = mo.accordion(
        lazy=True,
        items={
            "Files": files_display,
            "Storage": storage_display,
            "AIPs": aip_display,
            "Digitized vs born-digital content": digitized_born_digital_display,
            "Image vs AV": image_av_display,
            "Original vs duplicate files": original_duplicate_display,
        },
    )

    mo.vstack(
        ["Summary", summary, accordion],
        gap=1,
    )
    return


if __name__ == "__main__":
    app.run()
