# /// script
# requires-python = "==3.11.*"
# dependencies = [
#   "csd-python-api>=3.7.0",
#   "polars",
#   "jsonargparse",
#   "tqdm",
# ]
#
# [[tool.uv.index]]
# name = "ccdc"
# url = "https://pip.ccdc.cam.ac.uk/"
#
# [tool.uv.sources]
# csd-python-api = { index = "ccdc" }
# ///
import pathlib

import polars as pl
from ccdc.io import EntryReader
from jsonargparse import auto_cli
from tqdm import tqdm


def main(out: str = str(pathlib.Path("data/raw/csd_metadata.parquet"))):
    csd_reader = EntryReader("CSD")

    rows = []
    for entry in tqdm(csd_reader.entries(), total=len(csd_reader), desc="Exporting CSD metadata"):
        rows.append(
            {
                "id": entry.identifier,
                "deposition_date": entry.deposition_date,
                "r_factor": entry.r_factor,
                "has_3d_structure": entry.has_3d_structure,
                "has_disorder": entry.has_disorder,
                "is_powder_study": entry.is_powder_study,
                "pressure": entry.pressure,
                "is_polymeric": entry.is_polymeric,
            }
        )
    df = pl.from_dicts(
        rows,
        schema={
            "id": pl.String,
            "deposition_date": pl.Date,
            "r_factor": pl.Float64,
            "has_3d_structure": pl.Boolean,
            "has_disorder": pl.Boolean,
            "is_powder_study": pl.Boolean,
            "pressure": pl.String,
            "is_polymeric": pl.Boolean,
        },
    )
    df.write_parquet(out)


if __name__ == "__main__":
    auto_cli(main)
