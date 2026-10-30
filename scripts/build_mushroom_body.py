"""Extract the mushroom-body learning circuit for the plastic memory (FlyBrainLM mb=True).

In the fly, Kenyon cells (KCs) carry a sparse code of what is sensed; their synapses onto the mushroom-body output
neurons (MBONs) are rewired within seconds by dopamine from the DANs (PAM/PPL1) that innervate the same compartment.
Here every KC→MBON pair with ≥ MB_MIN_SYN synapses becomes a plastic edge (weak synapses kept: KC→MBON contacts are
individually small, so the brain-wide ≥5 threshold drops most of them), and each MBON's dopamine signal comes through
the real DAN→MBON wiring.

Output: data/old/connectomes/processed/{female,male}_mb.pt with neuron indices into {sex}_full.pt / {sex}_central.pt:
  {region: {kc, mbon, syn}}   plastic KC→MBON edges
  {region: {dan, dan_mbon, dan_syn}}   DAN→MBON teaching edges
"""
import json
import os
import re
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_connectome import OUT, ROOT  # noqa: E402

MB_MIN_SYN = 2
DAN_MIN_SYN = 2


def raw_edges(sex):
    if sex == "female":
        e = pd.read_feather(os.path.join(ROOT, "flywire_783", "proofread_connections_783.feather"),
                            columns=["pre_pt_root_id", "post_pt_root_id", "syn_count"])
        e = e.groupby(["pre_pt_root_id", "post_pt_root_id"], sort=False).syn_count.sum().reset_index()
    else:
        e = pd.read_feather(os.path.join(ROOT, "male_cns_v1.0", "connectome-weights-male-cns-v1.0-minconf-0.5-traced-only.feather"),
                            columns=["body_pre", "body_post", "weight"])
    e.columns = ["pre", "post", "syn"]
    return e


def main():
    summary = {"mb_min_syn": MB_MIN_SYN, "dan_min_syn": DAN_MIN_SYN}
    for sex in ("female", "male"):
        graphs = {r: torch.load(os.path.join(OUT, f"{sex}_{r}.pt")) for r in ("full", "central")}
        g = graphs["full"]
        types = np.array(g["neuron_type"])
        ids = np.array(g["ids"])
        kc_ids = set(ids[np.char.startswith(types.astype(str), "KC")].tolist())
        mbon_ids = set(ids[np.char.startswith(types.astype(str), "MBON")].tolist())
        dan_ids = set(ids[[bool(re.match(r"^(PAM|PPL1)", t)) for t in types]].tolist())
        e = raw_edges(sex)
        e = e[e.post.isin(mbon_ids)]
        km = e[e.pre.isin(kc_ids) & (e.syn >= MB_MIN_SYN)]
        dm = e[e.pre.isin(dan_ids) & (e.syn >= DAN_MIN_SYN)]
        out = {}
        for region, gr in graphs.items():
            index = pd.Series(np.arange(gr["n"]), index=gr["ids"])
            k = km[km.pre.isin(index.index) & km.post.isin(index.index)]
            d = dm[dm.pre.isin(index.index) & dm.post.isin(index.index)]
            out[region] = {
                "kc": torch.from_numpy(index[k.pre.to_numpy()].to_numpy().astype(np.int64)),
                "mbon": torch.from_numpy(index[k.post.to_numpy()].to_numpy().astype(np.int64)),
                "syn": torch.from_numpy(k.syn.to_numpy().astype(np.float32)),
                "dan": torch.from_numpy(index[d.pre.to_numpy()].to_numpy().astype(np.int64)),
                "dan_mbon": torch.from_numpy(index[d.post.to_numpy()].to_numpy().astype(np.int64)),
                "dan_syn": torch.from_numpy(d.syn.to_numpy().astype(np.float32)),
            }
            o = out[region]
            summary[f"{sex}_{region}"] = {
                "plastic_edges": len(o["kc"]), "kcs": len(set(o["kc"].tolist())), "mbons": len(set(o["mbon"].tolist())),
                "dan_edges": len(o["dan"]), "dans": len(set(o["dan"].tolist())),
                "mbons_with_dopamine": len(set(o["dan_mbon"].tolist()) & set(o["mbon"].tolist())),
            }
            print(sex, region, json.dumps(summary[f"{sex}_{region}"]))
        torch.save(out, os.path.join(OUT, f"{sex}_mb.pt"))
    with open(os.path.join(OUT, "mb_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
