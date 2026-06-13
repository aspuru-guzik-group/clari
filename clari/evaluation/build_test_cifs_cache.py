from __future__ import annotations

import polars as pl

from clari.csd import AVAILABLE_CSD_SUBSETS
from clari.datamodules import CrystalDataModule
from clari.paths import DATA_DIR


def main() -> None:
    fam_to_subsets: dict[str, list[str]] = {}
    for subset_name, cids in AVAILABLE_CSD_SUBSETS.items():
        for cid in cids:
            fam_to_subsets.setdefault(cid[:6], []).append(subset_name)

    split_opts = {"test": dict(group_by_fam=False, random_repr=False, augment=False)}
    dm = CrystalDataModule(batch_size=1, num_workers=0, collate_fn=None, split_opts=split_opts)

    rows = []
    for batch in dm.test_dataloader():
        crystal = batch.unbatch()[0]
        cid = str(crystal.csd_id)
        rows.append(
            {
                "csd_id": cid,
                "family": cid[:6],
                "cif": crystal.to_cif(wrap=False),
                "subsets": fam_to_subsets.get(cid[:6], []),
            }
        )

    out_path = DATA_DIR / "csd" / "test_cifs.parquet"
    pl.DataFrame(rows).write_parquet(out_path)
    print(f"Wrote {len(rows)} GT CIFs to {out_path}")


if __name__ == "__main__":
    main()
