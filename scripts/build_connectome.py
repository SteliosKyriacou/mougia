"""Turn the raw connectome tables into brain-only graphs for Kyriakou (female, FlyWire 783) and Markos (male CNS v1.0).

Output: data/old/connectomes/processed/{female,male}_{full,central}.pt, each a dict with
  n                 number of neurons
  pre, post         int64 edge lists (neuron indices), one edge per connected pair with >= MIN_SYN synapses
  syn               float32 synapse counts per edge
  sign              float32 per-neuron sign from the predicted neurotransmitter (+1 excitatory, -1 inhibitory)
  input_groups      {ORN type: [neuron indices]}  — the 53 olfactory receptor types (shared by both sexes)
  output_groups     {DN type: [neuron indices]}   — descending neuron types present in BOTH sexes
  neuron_type, superclass, ids  per-neuron metadata
"full" = whole brain incl. optic lobes; "central" = without optic-lobe intrinsic / visual neurons (fast pilots).
The input/output type lists are identical across sexes so only the wiring in between differs.
"""
import collections
import json
import os

import numpy as np
import pandas as pd
import torch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "old", "connectomes")
OUT = os.path.join(ROOT, "processed")
MIN_SYN = 5  # FlyWire's standard threshold for a reliable connection
INHIBITORY = {"gaba", "glutamate", "histamine"}  # glutamate acts mainly via GluCl (inhibitory) in the fly CNS

MALE_BRAIN = {"ol_intrinsic", "cb_intrinsic", "visual_projection", "ol_sensory", "cb_sensory", "ascending_neuron",
              "descending_neuron", "visual_centrifugal", "sensory_ascending", "cb_motor", "cb_endocrine",
              "sensory_descending", "cb_efferent", "efferent_ascending", "efferent_descending"}
OPTIC_MALE = {"ol_intrinsic", "visual_projection", "visual_centrifugal", "ol_sensory"}
OPTIC_FEMALE = {"optic", "visual_projection", "visual_centrifugal"}


def male():
    d = os.path.join(ROOT, "male_cns_v1.0")
    ann = pd.read_feather(os.path.join(d, "body-annotations-male-cns-v1.0-minconf-0.5.feather"))
    ann = ann[ann.superclass.isin(MALE_BRAIN)].copy()
    loc = ann.somaLocation.where(ann.somaLocation.notna(), ann.tosomaLocation)
    pos = np.array([np.asarray(v, float) * 8.0 if v is not None and len(v) == 3 else [np.nan] * 3 for v in loc])
    nt = pd.read_feather(os.path.join(d, "body-neurotransmitters-male-cns-v1.0.feather"))
    nt["nt"] = nt.consensus_nt.where(nt.consensus_nt != "unclear", nt.celltype_predicted_nt)
    ann = ann.merge(nt[["body", "nt"]], left_on="bodyId", right_on="body", how="left")
    w = pd.read_feather(os.path.join(d, "connectome-weights-male-cns-v1.0-minconf-0.5-traced-only.feather"),
                        columns=["body_pre", "body_post", "weight"])
    ann["cls"] = ann["class"]
    return dict(ids=ann.bodyId.to_numpy(), type=ann.type.fillna("").astype(str).to_numpy(),
                fwtype=ann.flywireType.fillna(ann.type).fillna("").astype(str).to_numpy(),
                superclass=ann.superclass.astype(str).to_numpy(), cls=ann.cls.fillna("").astype(str).to_numpy(),
                nt=ann.nt.fillna("").astype(str).str.lower().to_numpy(),
                edges=w.rename(columns={"body_pre": "pre", "body_post": "post", "weight": "syn"}),
                optic=ann.superclass.isin(OPTIC_MALE).to_numpy(),
                orn=(ann.cls == "olfactory").to_numpy() & ann.type.astype(str).str.startswith("ORN").to_numpy(),
                dn=(ann.superclass == "descending_neuron").to_numpy(), pos=pos,
                jo=ann.type.astype(str).str.startswith("JO").to_numpy())


def female():
    d = os.path.join(ROOT, "flywire_783")
    ann = pd.read_csv(os.path.join(d, "Supplemental_file1_neuron_annotations.tsv"), sep="\t", low_memory=False)
    conn = pd.read_feather(os.path.join(d, "proofread_connections_783.feather"), columns=["pre_pt_root_id", "post_pt_root_id", "syn_count"])
    edges = conn.groupby(["pre_pt_root_id", "post_pt_root_id"], sort=False).syn_count.sum().reset_index()
    edges.columns = ["pre", "post", "syn"]
    t = ann.cell_type.fillna(ann.hemibrain_type).fillna("").astype(str)
    return dict(ids=ann.root_id.to_numpy(), type=t.to_numpy(), fwtype=t.to_numpy(),
                superclass=ann.super_class.astype(str).to_numpy(), cls=ann.cell_class.fillna("").astype(str).to_numpy(),
                nt=ann.top_nt.fillna("").astype(str).str.lower().to_numpy(), edges=edges,
                optic=ann.super_class.isin(OPTIC_FEMALE).to_numpy(),
                orn=(ann.cell_class == "olfactory").to_numpy() & t.str.startswith("ORN").to_numpy(),
                dn=(ann.super_class == "descending").to_numpy(),
                jo=t.str.startswith("JO").to_numpy(),
                pos=ann[["pos_x", "pos_y", "pos_z"]].to_numpy(float) * np.array([4.0, 4.0, 40.0]))


def fill_positions(pos, pre, post, syn, rounds=5):
    """Neurons without a recorded position (e.g. sensory neurons with no soma in the brain) are placed at the
    synapse-weighted mean position of their partners, repeated a few times; any left over go to the centroid."""
    pos = pos.copy()
    missing = np.isnan(pos).any(1)
    estimated = missing.copy()
    a = np.concatenate([pre, post]); b = np.concatenate([post, pre]); w = np.concatenate([syn, syn]).astype(float)
    for _ in range(rounds):
        if not missing.any():
            break
        known = ~missing[b] & missing[a]
        acc = np.zeros((len(pos), 3)); tot = np.zeros(len(pos))
        np.add.at(acc, a[known], pos[b[known]] * w[known, None])
        np.add.at(tot, a[known], w[known])
        fill = missing & (tot > 0)
        pos[fill] = acc[fill] / tot[fill, None]
        missing &= ~fill
    pos[missing] = np.nanmean(pos[~missing], 0) if (~missing).any() else 0
    return pos, estimated


def build(g, keep, in_types, out_types, audio_types=(), fair=None):
    ids = g["ids"][keep]
    index = pd.Series(np.arange(len(ids)), index=ids)
    e = g["edges"]
    e = e[(e.syn >= MIN_SYN) & e.pre.isin(index.index) & e.post.isin(index.index) & (e.pre != e.post)]
    pre, post = index[e.pre.to_numpy()].to_numpy(), index[e.post.to_numpy()].to_numpy()
    nt = g["nt"][keep]
    sign = np.where(np.isin(nt, list(INHIBITORY)), -1.0, 1.0).astype(np.float32)
    ftype = g["fwtype"][keep]
    pos, estimated = fill_positions(g["pos"][keep], pre, post, e.syn.to_numpy())
    groups = lambda mask, types: {t: np.flatnonzero(mask & (ftype == t)).tolist() for t in types}
    d = {
        "n": int(len(ids)), "pre": torch.from_numpy(pre.astype(np.int64)), "post": torch.from_numpy(post.astype(np.int64)),
        "syn": torch.from_numpy(e.syn.to_numpy().astype(np.float32)), "sign": torch.from_numpy(sign),
        "input_groups": groups(g["orn"][keep], in_types), "output_groups": groups(g["dn"][keep], out_types),
        "audio_groups": groups(g["jo"][keep], audio_types),
        "neuron_type": ftype.tolist(), "superclass": g["superclass"][keep].tolist(), "ids": ids.tolist(),
        "nt": nt.tolist(), "min_syn": MIN_SYN,
        "pos": torch.from_numpy(pos.astype(np.float32)), "pos_estimated": torch.from_numpy(estimated),
    }
    # Fair per-neuron interface: per cell type, the same number of neurons in both sexes (the smaller count),
    # chosen deterministically (lowest ids). Order = type order, so channel k means the same cell type in both flies.
    if fair:
        for name, key in (("input_neurons", "input_groups"), ("audio_neurons", "audio_groups"), ("output_neurons", "output_groups")):
            idx = []
            for t in sorted(d[key]):
                members = sorted(d[key][t], key=lambda i: ids[i])
                idx += members[: fair[key].get(t, 0)]
            d[name] = idx
    return d


def main():
    os.makedirs(OUT, exist_ok=True)
    flies = {"female": female(), "male": male()}
    # Shared interface: types present in both sexes.
    in_types = sorted(set.intersection(*[set(g["fwtype"][g["orn"]]) for g in flies.values()]) - {""})
    out_types = sorted(set.intersection(*[set(g["fwtype"][g["dn"]]) for g in flies.values()]) - {""})
    audio_types = sorted(set.intersection(*[set(g["fwtype"][g["jo"]]) for g in flies.values()]) - {""})
    # per-type neuron counts in each sex (whole brain), fair count = the smaller one
    fair = {}
    for key, mask, types in (("input_groups", "orn", in_types), ("audio_groups", "jo", audio_types), ("output_groups", "dn", out_types)):
        counts = [{t: int(((g["fwtype"] == t) & g[mask]).sum()) for t in types} for g in flies.values()]
        fair[key] = {t: min(c[t] for c in counts) for t in types}
    summary = {"input_types": len(in_types), "output_types": len(out_types), "audio_types": len(audio_types), "min_syn": MIN_SYN}
    for sex, g in flies.items():
        for region in ("full", "central"):
            keep = np.ones(len(g["ids"]), bool) if region == "full" else ~g["optic"]
            d = build(g, keep, in_types, out_types, audio_types, fair)
            torch.save(d, os.path.join(OUT, f"{sex}_{region}.pt"))
            summary[f"{sex}_{region}"] = {
                "neurons": d["n"], "edges": int(len(d["pre"])), "synapses": int(d["syn"].sum()),
                "inhibitory_frac": round(float((d["sign"] < 0).float().mean()), 3),
                "input_neurons": sum(map(len, d["input_groups"].values())), "output_neurons": sum(map(len, d["output_groups"].values())),
                "positions_estimated": int(d["pos_estimated"].sum()),
                "fair_input_neurons": len(d["input_neurons"]), "fair_audio_neurons": len(d["audio_neurons"]),
                "fair_output_neurons": len(d["output_neurons"]),
                "superclasses": dict(collections.Counter(d["superclass"]).most_common(8)),
            }
            print(sex, region, json.dumps(summary[f"{sex}_{region}"]))
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({k: summary[k] for k in ("input_types", "output_types", "audio_types")}))


if __name__ == "__main__":
    main()
