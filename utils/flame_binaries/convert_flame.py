#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pathlib import Path
import pickle

import numpy as np
import torch

_cwd = Path(__file__).parent.resolve()


def main():
    uv = torch.load(_cwd / "uv.pt")
    for name in ["male_model", "female_model"]:
        path = _cwd / f"{name}.pkl"
        with open(path, "rb") as f:
            di = pickle.load(f, encoding="latin1")

        df = {
            "T": torch.from_numpy(di["v_template"]).float(),
            "faces": torch.from_numpy(di["f"].astype(np.int64)).long(),
            "S": torch.from_numpy(di["shapedirs"].x[..., :300]).float(),
            "E": torch.from_numpy(di["shapedirs"].x[..., 300:]).float(),
            "P": torch.from_numpy(di["posedirs"]).float(),
            "W": torch.from_numpy(di["weights"]).float(),
            "parents": torch.from_numpy(di["kintree_table"][0]).long(),
            "J": torch.from_numpy(np.array(di["J_regressor"].todense())).float(),
            "uvfaces": uv["uvfaces"],
            "uvcoords": uv["uvcoords"],
        }
        df["parents"][0] = -1

        torch.save(df, path.with_suffix(".pt"))


if __name__ == "__main__":
    main()
