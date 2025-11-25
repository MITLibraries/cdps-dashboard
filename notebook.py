# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "boto3",
#     "marimo",
#     "pandas",
#     "plotly",
#     "pyarrow",
# ]
# ///

import marimo

__generated_with = "0.18.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(
        """
    # CDPS Dashboard
    """
    )
    return


@app.cell
def _():
    # Functions
    import io
    import logging
    import math
    import mimetypes
    import os
    import re
    from datetime import datetime, timedelta
    from pathlib import Path
    from urllib.parse import urlparse

    import boto3
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    from botocore.exceptions import ClientError

    logger = logging.getLogger(__name__)
    logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s")
    logger.setLevel(logging.INFO)

    def convert_size(size_bytes):
        """Convert byte counts into a human readable format."""
        if size_bytes == 0:
            return "0B"
        size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
        i = math.floor(math.log(size_bytes, 1024))
        p = math.pow(1024, i)
        s = round(size_bytes / p, 2)
        return f"{s} {size_name[i]}"

    def rename_bucket(dataframe: pd.DataFrame) -> pd.DataFrame:
        """Extract AIPStore name from bucket field (e.g., 'aipstore1b')."""
        dataframe.loc[:, "bucket"] = dataframe["bucket"].str.extract(
            r"(aipstore\d+[a-z]?)", expand=False
        )
        return dataframe

    def parse_s3_keys(dataframe: pd.DataFrame) -> pd.DataFrame:
        """Parse S3 keys to extract additional metadata."""
        key_parts = dataframe["key"].str.split("/", expand=True)

        dataframe.loc[:, "bagname"] = key_parts[8] if key_parts.shape[1] > 8 else ""
        dataframe.loc[:, "uuid"] = (
            key_parts[0]
            + key_parts[1]
            + "-"
            + key_parts[2]
            + "-"
            + key_parts[3]
            + "-"
            + key_parts[4]
            + "-"
            + key_parts[5]
            + key_parts[6]
            if key_parts.shape[1] > 6
            else "" + key_parts[7] if key_parts.shape[1] > 7 else ""
        )
        dataframe["accession_name"] = dataframe["bagname"].str.split("-").str[0]
        dataframe.loc[:, "file"] = dataframe["key"].str.split("/").str[-1]
        dataframe.loc[:, "filepath"] = (
            dataframe["key"].str.split("/").str[9:].apply("/".join)
        )

        dataframe.loc[:, "extension"] = dataframe["key"].apply(
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

    def preservation_level(dataframe: pd.DataFrame) -> pd.DataFrame:
        """Add preservation level based on S3 bucket."""
        dataframe.loc[:, "preservation_level"] = np.where(
            dataframe["bucket"].str.contains("1b"),
            "Level 1",
            np.where(
                dataframe["bucket"].str.contains("2b"),
                "Level 2",
                np.where(
                    dataframe["bucket"].str.contains("3b"),
                    "Level 3",
                    np.where(
                        dataframe["bucket"].str.contains("4b")
                        | dataframe["bucket"].str.contains("4a"),
                        "Level 4",
                        np.where(
                            dataframe["bucket"].str.contains("5b")
                            | dataframe["bucket"].str.contains("5a"),
                            "Level 5",
                            "ERROR",
                        ),
                    ),
                ),
            ),
        )
        return dataframe

    def mime_types(dataframe: pd.DataFrame) -> pd.DataFrame:
        """Add mime type based on file extension."""
        mimetypes.add_type("application/vnd.ms-outlook", ".msg")
        dataframe.loc[:, "mimetype"] = dataframe["extension"].apply(
            lambda extension: (
                "unknown"
                if pd.isna(extension) or not extension
                else mimetypes.types_map.get(extension, "unknown")
            )
        )
        return dataframe

    def is_digitized_aip(dataframe: pd.DataFrame) -> pd.DataFrame:
        """Identifies digitized AIPS based on UUID."""
        digitized_aip_regex = r"\d{4}_\d{3}[r|R]{2}_\d{3}"
        dataframe.loc[:, "is_digitized_AIP"] = np.where(
            dataframe.accession_name.str.contains(digitized_aip_regex, regex=True),
            "Digitized",
            np.where(
                dataframe.accession_name.isin(os.environ["DIGITIZED_BAG_IDS"].split(",")),
                "Digitized",
                "Born Digital",
            ),
        )
        return dataframe

    def is_replica(dataframe: pd.DataFrame) -> pd.DataFrame:
        """Identifies replicas based on S3 bucket."""
        dataframe.loc[:, "is_replica"] = np.where(
            dataframe["bucket"].str.contains("4b"),
            True,
            np.where(dataframe["bucket"].str.contains("5b"), True, False),
        )
        return dataframe

    def is_normalized_file(dataframe: pd.DataFrame) -> pd.DataFrame:
        """Identifies normalized files based on several criteria."""
        am_uuid_regex = (
            r"-\S{8}-\S{4}-\S{4}-\S{4}-\S{12}."  # regex for archivematica file UUID
        )
        dataframe.loc[:, "is_normalized_file"] = np.where(
            dataframe.file.str.contains(am_uuid_regex, regex=True),
            True,
            np.where(
                dataframe.file.str.contains("data/thumbnails"),
                True,
                np.where(
                    (dataframe["is_digitized_AIP"] == "Digitized")
                    & (dataframe.file.str.contains(".pdf")),
                    True,
                    False,
                ),
            ),
        )
        return dataframe

    def set_status(dataframe: pd.DataFrame) -> pd.DataFrame:
        """Adds a status based on CDPS content categories."""
        dataframe.loc[:, "status"] = np.where(
            dataframe["is_replica"],
            "replica",
            np.where(
                dataframe["is_normalized_file"],
                "normalized/access",
                np.where(dataframe["is_metadata"], "metadata", "original content"),
            ),
        )
        return dataframe

    return (
        ClientError,
        boto3,
        convert_size,
        datetime,
        go,
        io,
        is_digitized_aip,
        is_metadata,
        is_normalized_file,
        is_replica,
        logger,
        mime_types,
        os,
        parse_s3_keys,
        pd,
        preservation_level,
        re,
        rename_bucket,
        set_status,
        timedelta,
        urlparse,
    )


@app.cell
def _(ClientError, boto3, logger, os, re, urlparse):
    # Get symlink files and dates as dict
    s3 = boto3.client("s3")

    logger.info("Building symlink dict from S3 inventory locations")
    symlink_dict = {}

    # Iterate through the S3 inventory locations and build symlink dict
    for s3_inventory_location in os.environ["S3_INVENTORY_LOCATIONS"].split(","):
        logger.info(f"Retrieving symlink.txt files from: {s3_inventory_location}")
        parsed_location = urlparse(s3_inventory_location)
        inventory_bucket = parsed_location.netloc
        inventory_prefix = parsed_location.path.lstrip("/")

        paginator = s3.get_paginator("list_objects_v2")
        try:
            for page in paginator.paginate(
                Bucket=inventory_bucket, Prefix=inventory_prefix
            ):
                for obj in page.get("Contents", []):
                    key = obj["Key"]
                    if key.lower().endswith("symlink.txt"):
                        if match := re.search(r"dt=\d{4}-\d{2}-\d{2}", key):
                            date_string = match.group(0)[3:]
                        else:
                            raise ValueError(
                                f"Could not parse datetime partition from uri: {key}"
                            )

                        if date_string not in symlink_dict:
                            symlink_dict[date_string] = []
                        symlink_dict[date_string].append(f"s3://{inventory_bucket}/{key}")
        except ClientError:
            logger.exception("Client error while retrieving symlink.txt files:")
            raise
    logger.info(f"Symlink dict built with {len(symlink_dict)} dates.")
    return s3, symlink_dict


@app.cell
def _(datetime, mo, timedelta):
    # Select date from calendar element

    yesterday = (datetime.now() - timedelta(days=1)).date()
    date_selector = mo.ui.date(value=str(yesterday), label="Select S3 Inventory Date")
    date_selector
    return (date_selector,)


@app.cell
def _(date_selector, pd):
    # Retrieve parquet files from the selected date

    selected_date = pd.to_datetime(date_selector.value).strftime("%Y-%m-%d")
    return (selected_date,)


@app.cell
def _():
    # Cache of parquet files URIs by date for efficient recall of previously used dates

    parquet_file_uri_cache = {}
    return (parquet_file_uri_cache,)


@app.cell
def _(
    ClientError,
    io,
    logger,
    parquet_file_uri_cache,
    pd,
    s3,
    selected_date,
    symlink_dict,
    urlparse,
):
    # Add parquet file URIs to cache if not already present
    if not parquet_file_uri_cache.get(selected_date):
        logger.info(f"Collecting parquet file URIs for date: {selected_date}")
        parquet_file_uris = []
        for symlink in symlink_dict[selected_date]:
            # Get parquet file URI from symlink.txt
            parsed_symlink_file = urlparse(symlink)
            symlink_bucket = parsed_symlink_file.netloc
            symlink_key = parsed_symlink_file.path.lstrip("/")
            try:
                logger.info(
                    f"Retrieving symlink file: s3://{symlink_bucket}/{symlink_key}"
                )
                response = s3.get_object(Bucket=symlink_bucket, Key=symlink_key)
            except ClientError:
                logger.exception("Client error while retrieving symlink.txt file:")
                raise
            parquet_file_uris.append(response["Body"].read().decode("utf-8"))
        parquet_file_uri_cache[selected_date] = parquet_file_uris

    # Retrieve parquet files
    parquet_dfs = []
    logger.info(f"Processing parquet file URIs for date: {selected_date}")
    for parquet_file_uri in parquet_file_uri_cache[selected_date]:
        # Parse parquet file URI
        parsed_parquet_file_uri = urlparse(parquet_file_uri)
        parquet_bucket = parsed_parquet_file_uri.netloc
        parquet_key = parsed_parquet_file_uri.path.lstrip("/")

        # Get parquet file and convert to dataframe
        try:
            logger.info(f"Retrieving parquet file: s3://{parquet_bucket}/{parquet_key}")
            s3_object = s3.get_object(Bucket=parquet_bucket, Key=parquet_key)
        except ClientError:
            logger.exception("Client error while retrieving parquet file:")
            raise
        parquet_df = pd.read_parquet(io.BytesIO(s3_object["Body"].read()))
        parquet_dfs.append(parquet_df)

    # Concatenate parquet dataframes
    inventory_df = (
        pd.concat(parquet_dfs, ignore_index=True).drop_duplicates().reset_index(drop=True)
    )

    # Keep only current objects in dataframe
    inventory_df.loc[:, "is_current"] = (
        inventory_df["is_latest"] & ~inventory_df["is_delete_marker"]
    )
    current_df = (
        inventory_df.loc[inventory_df["is_current"]].copy().reset_index(drop=True)
    )
    logger.info(f"Current CDPS dataframe built with {len(current_df)} records.")
    return (current_df,)


@app.cell
def _(
    current_df,
    is_digitized_aip,
    is_metadata,
    is_normalized_file,
    is_replica,
    mime_types,
    parse_s3_keys,
    preservation_level,
    rename_bucket,
    set_status,
):
    # Update dataframe with additional metadata
    cdps_df = (
        current_df.pipe(rename_bucket)
        .pipe(parse_s3_keys)
        .pipe(is_metadata)
        .pipe(preservation_level)
        .pipe(mime_types)
        .pipe(is_digitized_aip)
        .pipe(is_replica)
        .pipe(is_normalized_file)
        .pipe(set_status)
    )
    return (cdps_df,)


@app.cell
def _(cdps_df, go, mo):
    # Files

    # Data views generated from filtered dataframes
    file_bucket = (
        cdps_df.groupby("bucket")
        .size()
        .to_frame("file count")
        .sort_values(by="bucket", ascending=False)
    )
    file_status = (
        cdps_df.groupby("status")
        .size()
        .to_frame("file count")
        .sort_values(by="status", ascending=False)
    )
    file_bucket_status = (
        cdps_df.groupby(["bucket", "status"])
        .size()
        .to_frame("file count")
        .sort_values(by="bucket", ascending=False)
    )
    file_preservation = (
        cdps_df.groupby("preservation_level")
        .size()
        .to_frame("file count")
        .sort_values(by="preservation_level", ascending=False)
    )

    # Create pie chart for presevation level
    preservation_data = file_preservation.reset_index()
    preservation_chart = go.Figure(
        data=[
            go.Pie(
                labels=preservation_data["preservation_level"],
                values=preservation_data["file count"],
                title="File count by preservation level",
            )
        ]
    )
    preservation_chart.update_layout(height=400, width=500)

    # Organizes the data views into tables vertically with labels
    files_display = mo.vstack(
        [
            mo.md("#### File count by bucket"),
            mo.ui.table(file_bucket),
            mo.md("#### File count by status"),
            mo.ui.table(file_status),
            mo.md("#### File count by bucket and status"),
            mo.ui.table(file_bucket_status),
            mo.md("#### File count by preservation level"),
            mo.ui.plotly(preservation_chart),
            mo.ui.table(file_preservation),
        ],
        gap=1,
    )
    return (files_display,)


@app.cell
def _(mo):
    # Storage

    # Data views generated from filtered dataframes
    storage = {"not implemented": "not implemented"}

    # Organizes the data views into tables vertically with labels
    storage_display = mo.vstack(
        [mo.ui.table(storage)],
        gap=1,
    )
    return (storage_display,)


@app.cell
def _(cdps_df, mo):
    # File extension and mimetypes

    _file_extensions = (
        cdps_df.groupby("extension")
        .size()
        .to_frame("file count")
        .sort_values(by="file count", ascending=False)
    )

    # Data views generated from filtered dataframes
    file_extensions_mimetypes = {"not implemented": "not implemented"}

    # Organizes the data views into tables vertically with labels
    file_extensions_mimetypes_display = mo.vstack(
        [mo.ui.table(file_extensions_mimetypes)],
        gap=1,
    )
    return (file_extensions_mimetypes_display,)


@app.cell
def _(cdps_df, convert_size, mo):
    # File datapoints

    # Data views generated from filtered dataframes
    _file_metadata = (
        cdps_df.groupby("is_metadata")
        .size()
        .rename(index={False: "content files", True: "metadata files"})
        .to_frame("file count")
    )

    _file_storage = (
        cdps_df.sort_values(by="size", ascending=False)
        .loc[:, ["file", "size"]]
        .assign(size=lambda x: x["size"].apply(convert_size))
        .reset_index(drop=True)[:10]
    )

    file_datapoints = {"not implemented": "not implemented"}

    # Organizes the data views into tables vertically with labels
    file_datapoints_display = mo.vstack(
        [mo.ui.table(file_datapoints)],
        gap=1,
    )
    return (file_datapoints_display,)


@app.cell
def _(mo):
    # AIPs

    # Data views generated from filtered dataframes
    aips = {"not implemented": "not implemented"}

    # Organizes the data views into tables vertically with labels
    aip_display = mo.vstack(
        [mo.ui.table(aips)],
        gap=1,
    )
    return (aip_display,)


@app.cell
def _(mo):
    # Digitized vs born-digital content

    # Data views generated from filtered dataframes
    digitized_born_digital = {"not implemented": "not implemented"}

    # Organizes the data views into tables vertically with labels
    digitized_born_digital_display = mo.vstack(
        [mo.ui.table(digitized_born_digital)],
        gap=1,
    )
    return (digitized_born_digital_display,)


@app.cell
def _(mo):
    # Image vs AV

    # Data views generated from filtered dataframes
    image_av = {"not implemented": "not implemented"}

    # Organizes the data views into tables vertically with labels
    image_av_display = mo.vstack(
        [mo.ui.table(image_av)],
        gap=1,
    )
    return (image_av_display,)


@app.cell
def _(mo):
    # Original vs duplicate files

    # Data views generated from filtered dataframes
    original_duplicate = {"not implemented": "not implemented"}

    # Organizes the data views into tables vertically with labels
    original_duplicate_display = mo.vstack(
        [mo.ui.table(original_duplicate)],
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
        label="Total storage size",
        value=f"{convert_size(cdps_df["size"].sum())}",
    )

    # Organizes the summary stats horizontally
    current_summary = mo.hstack([total_files, total_storage], widths="equal", gap=1)
    return (current_summary,)


@app.cell
def _(
    aip_display,
    current_summary,
    digitized_born_digital_display,
    file_datapoints_display,
    file_extensions_mimetypes_display,
    files_display,
    image_av_display,
    mo,
    original_duplicate_display,
    storage_display,
):
    # Dashboard

    # Collects all the data displays with labels in an accordion element
    accordion = mo.accordion(
        lazy=True,
        items={
            "Files": files_display,
            "Storage": storage_display,
            "File extensions and mimetypes": file_extensions_mimetypes_display,
            "File datapoints": file_datapoints_display,
            "AIPs": aip_display,
            "Digitized vs born-digital content": digitized_born_digital_display,
            "Image vs AV": image_av_display,
            "Original vs duplicate files": original_duplicate_display,
        },
    )

    # Organizes elements on the page vertically
    mo.vstack(
        [mo.md("### Summary"), current_summary, accordion],
        gap=1,
    )
    return


if __name__ == "__main__":
    app.run()
