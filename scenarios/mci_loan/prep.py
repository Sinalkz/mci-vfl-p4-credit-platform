"""MCI loan instance: the prepared inputs, in data/recs/prep.py's format (the loader is scenarios/mci_loan/offchain/prepared.mjs).

- Rows: the cohort (common.py: 15,965 applicants of application_train, balanced on TARGET), in SK_ID_CURR order.
- Label: TARGET (1 = payment difficulties, "default"; 0 = repaid). Classes 2-4 of the thesis's 5-class structure stay
  empty (docs/mci/00-recon.md section 5, point 3).
- Bank (active): its 8 columns (results/mci_loan/select_bank.json), one number each (bank.py; WoE out of fold on the
  training rows), missing values set to the training median, standardized with the training mean and SD, times 2^12,
  rounded, clipped to int16 (clipping counted).
- Passives: party_rule.PASSIVE_COLUMNS from features.py: transform, standardized over the training rows that have a
  value, 0 where there is none (the neutral fill, decision (c)), clipped to +-5; 0/1 indicators as they are. A column
  constant on the training rows is dropped (as data/recs/prep.py).
- The extra pool: 2,531 applicants outside the cohort, passive columns only, encoded with the cohort's training
  statistics (the passives' applicants that the Bank does not have; Step 5's alignment drops them).
- Evaluator-side: thin_file = no row in bureau.csv (segment analysis only); the population default rate of
  application_train (the business metric's reweighting, decision (b)); the no-record counts per member and table.
- The smoke sample (Step 3): 1,000 cohort rows (common.smoke_positions), the same encoded values, in prepared-smoke/.

Outputs: data/mci_loan/prepared/ and prepared-smoke/ (git-ignored: a_int16.bin, passive_f32.bin, passive_extra_f32.bin,
manifest.json, extra_manifest.json, columns.json, README.md); copies of the JSON and README under results/mci_loan/.

Usage (from the repo root):  .venv/Scripts/python -I scenarios/mci_loan/prep.py
"""
from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import bank  # noqa: E402
import common as cm  # noqa: E402
import features  # noqa: E402
import party_rule as pr  # noqa: E402

CLIP_Z = 5.0
SCALE = 4096


def encode_bank(app_c: pd.DataFrame, y: np.ndarray, train_pos: np.ndarray, test_pos: np.ndarray, selected: list[str],
                raw_fn=bank.raw_candidates, kind_fn=None):
    """The active party's selected columns as int16 at 2^12. raw_fn, kind_fn: v2's (MCI's candidates); default v1's Bank."""
    raw = raw_fn(app_c)
    ids = app_c.index.to_numpy()
    fold_of = np.full(len(ids), -1)
    fold_of[train_pos] = bank.folds(ids[train_pos], y[train_pos], cm.FOLD_SEED)
    X, woe = bank.numeric_matrix(raw, y, train_pos, fold_of, selected)
    cols, info = [], {}
    for j, c in enumerate(selected):
        x = X[:, j].copy()
        med = float(np.nanmedian(x[train_pos]))
        missing = np.isnan(x)
        x[missing] = med
        mean, sd = float(x[train_pos].mean()), float(x[train_pos].std()) or 1.0
        z = (x - mean) / sd
        q = np.round(z * SCALE)
        over = (q < -32768) | (q > 32767)
        cols.append(np.clip(q, -32768, 32767).astype(np.int16))
        kind = kind_fn(c) if kind_fn else next((k for k, v in pr.BANK_RAW.items() if c in v), "derived")
        info[c] = {"kind": kind, "formula": pr.BANK_DERIVED[c][0] if c in pr.BANK_DERIVED else None,
                   "missing_train": int(missing[train_pos].sum()), "missing_test": int(missing[test_pos].sum()),
                   "train_median": med, "train_mean": mean, "train_sd": sd, "clipped_train": int(over[train_pos].sum()),
                   "clipped_test": int(over[test_pos].sum()), "zmin": float(z.min()), "zmax": float(z.max()),
                   **({"woe": woe[c]} if c in woe else {})}
    return np.column_stack(cols).astype("<i2"), info


def encode_passives(F: pd.DataFrame, train_rows: np.ndarray, passives=pr.PASSIVES, columns=pr.PASSIVE_COLUMNS):
    """F: raw passive columns for the cohort and then the pool; train_rows: F's rows of the cohort's training split;
    passives, columns: the party rule's (v2: scenarios/mci_loan/v2/rule2.py; default v1's).
    Returns the float32 matrix (all F rows), the kept columns' records and the dropped ones."""
    blocks, kept, dropped = [], [], []
    for owner in passives:
        for spec in (s for s in columns if s["owner"] == owner):
            v = F[spec["name"]].to_numpy(dtype=float)
            t = spec["transform"]
            if t == "log":
                v = np.log1p(np.clip(v, 0, None))
            elif t == "slog":
                v = np.sign(v) * np.log1p(np.abs(v))
            tr = v[train_rows]
            present = ~np.isnan(tr)
            rec = {k: spec[k] for k in ("name", "owner", "source", "transform", "cols", "meaning")}
            if t == "ind":
                if np.all(tr == tr[0]):
                    dropped.append({**rec, "reason": "constant on the training rows"})
                    continue
                out = v
                rec.update(train_share_one=float(tr.mean()))
            else:
                if present.sum() == 0 or np.nanstd(tr) == 0:
                    dropped.append({**rec, "reason": "constant (or empty) on the training rows"})
                    continue
                mean, sd = float(np.nanmean(tr)), float(np.nanstd(tr))
                z = (v - mean) / sd
                clipped = int((np.abs(z[train_rows]) > CLIP_Z).sum())
                out = np.clip(np.nan_to_num(z, nan=0.0), -CLIP_Z, CLIP_Z)
                rec.update(train_mean=mean, train_sd=sd, present_train=int(present.sum()), clipped_train=clipped)
            kept.append(rec)
            blocks.append(out)
    P = np.column_stack(blocks).astype("<f4")
    assert np.isfinite(P).all()
    for k, c in enumerate(kept):
        c["col"] = k
    return P, kept, dropped


def write_set(folder: Path, results: Path, A, P, ids, y, train_pos, test_pos, extra: dict) -> dict:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "a_int16.bin").write_bytes(A.tobytes())
    (folder / "passive_f32.bin").write_bytes(P.tobytes())
    m = {"rows": int(len(ids)), "ids": [int(i) for i in ids],
         "split": {"seed": cm.SPLIT_SEED, "train_pos": [int(p) for p in train_pos], "test_pos": [int(p) for p in test_pos]},
         "y": [int(v) for v in y],
         "class_counts": {"train": np.bincount(y[train_pos], minlength=2).tolist(), "test": np.bincount(y[test_pos], minlength=2).tolist()},
         "a": {**extra.pop("a"), "file": "a_int16.bin", "dtype": "int16 little-endian", "shape": list(A.shape),
               "sha256": cm.sha256_file(folder / "a_int16.bin")},
         "passive": {**extra.pop("passive"), "width": int(P.shape[1]), "file": "passive_f32.bin", "dtype": "float32 little-endian",
                     "shape": list(P.shape), "sha256": cm.sha256_file(folder / "passive_f32.bin")},
         **extra}
    cm.write_json(folder / "manifest.json", m)
    results.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(folder / "manifest.json", results / "manifest.json")
    return m


def readme(m: dict, cols: list, extra_m: dict | None) -> str:
    L = ["# MCI loan instance: prepared inputs", "", "Generated by `scenarios/mci_loan/prep.py`; do not edit by hand.", "",
         f"- Rows: {m['rows']:,} applicants (SK_ID_CURR order); training {len(m['split']['train_pos']):,}, test "
         f"{len(m['split']['test_pos']):,}.",
         f"- Classes (repaid / default): training {m['class_counts']['train'][0]:,} / {m['class_counts']['train'][1]:,}; test "
         f"{m['class_counts']['test'][0]:,} / {m['class_counts']['test'][1]:,}.",
         f"- thin_file (no bureau record): {len(m['thin_file_pos']):,} rows.",
         f"- Population default rate of application_train (reweighting): {m['population']['default_rate']:.6f} "
         f"({m['population']['defaults']:,} of {m['population']['rows']:,}).", "",
         "## Bank: 8 columns, int16 at 2^12", "", "| Column | Kind | Missing (train / test) | Training mean | Training SD | Clipped (train / test) |",
         "|---|---|---:|---:|---:|---:|"]
    for c in m["a"]["vars"]:
        i = m["a"]["info"][c]
        L.append(f"| {c} | {i['kind']} | {i['missing_train']} / {i['missing_test']} | {i['train_mean']:.4f} | {i['train_sd']:.4f} | "
                 f"{i['clipped_train']} / {i['clipped_test']} |")
    L += ["", "## Passives", "", "| Party | Columns |", "|---|---:|"]
    L += [f"| {p} | {len(v)} |" for p, v in m["parties"].items()]
    L += ["", f"Total width {m['passive']['width']}; dropped as constant on the training rows: "
          f"{', '.join(d['name'] for d in m['passive']['dropped']) or 'none'}.", "",
          "## No record (applicants with no row in the table)", "", "| Member | Table | Rows with no record |", "|---|---|---:|"]
    for t, n in m["no_record"]["by_table"].items():
        L.append(f"| {pr.TABLES[t]} | {t} | {n:,} |")
    L += [f"| {p} | any of its tables | {n:,} |" for p, n in m["no_record"]["by_member"].items()]
    if extra_m:
        L += ["", f"Extra pool: {extra_m['rows']:,} applicants outside the cohort, passive columns only "
              f"(`{extra_m['passive']['file']}`, SHA-256 `{extra_m['passive']['sha256']}`)."]
    L += ["", f"Files: `a_int16.bin` SHA-256 `{m['a']['sha256']}`; `passive_f32.bin` SHA-256 `{m['passive']['sha256']}`.", ""]
    return "\n".join(L)


def main() -> None:
    t0 = time.time()
    app = pd.read_csv(cm.RAW / "application_train.csv").set_index("SK_ID_CURR", drop=False)
    all_ids = app.index.to_numpy()
    target = app["TARGET"].to_numpy()
    cohort = cm.draw_cohort(all_ids, target)
    pool = cm.draw_extra_pool(all_ids, cohort)
    app_c = app.loc[cohort]
    y = app_c["TARGET"].to_numpy()
    train_pos, test_pos = cm.split_positions(cohort, y)
    smoke = cm.smoke_positions(cohort, y, train_pos, test_pos)
    sel = json.loads((cm.RESULTS / "select_bank.json").read_text(encoding="utf-8"))["selected"]
    assert len(sel) == pr.BANK_N_COLUMNS

    A, a_info = encode_bank(app_c, y, train_pos, test_pos, sel)
    F = features.passive_frame(app.loc[np.concatenate([cohort, pool])])
    P_all, kept, dropped = encode_passives(F, train_pos)
    P, P_extra = P_all[:len(cohort)], P_all[len(cohort):]
    parties = {p: [c["col"] for c in kept if c["owner"] == p] for p in pr.PASSIVES}
    Fc = F.iloc[:len(cohort)]
    by_table = {t: int((Fc[h] == 0).sum()) for t, h in pr.HAS_RECORD.items()}
    by_member = {p: int(np.all([Fc[pr.HAS_RECORD[t]] == 0 for t, o in pr.TABLES.items() if o == p], axis=0).sum()) for p in pr.PASSIVES}
    thin = np.flatnonzero(Fc["BUR_HAS_RECORD"].to_numpy() == 0)
    common = {
        "variables": {p: [c["name"] for c in kept if c["owner"] == p] for p in pr.PASSIVES},
        "population": {"rows": int(len(target)), "defaults": int(target.sum()), "default_rate": float(target.mean())},
        "cohort": {"seed": cm.COHORT_SEED, "n": cm.N_COHORT, "defaults": cm.N_DEFAULT, "repaid": cm.N_REPAID},
        "seeds": {"cohort": cm.COHORT_SEED, "split": cm.SPLIT_SEED, "extra": cm.EXTRA_SEED, "smoke": cm.SMOKE_SEED, "folds": cm.FOLD_SEED},
        "label": {"0": "repaid (TARGET 0)", "1": "default: payment difficulties (TARGET 1)"},
        "slots": json.loads(cm.SLOTS.read_text(encoding="utf-8")),
    }
    def extra_fields(rows):
        return {"a": {"vars": sel, "scale": SCALE, "info": a_info}, "passive": {"dropped": dropped}, "parties": parties,
                "thin_file_pos": [int(p) for p in rows], **common}

    full = write_set(cm.PREPARED, cm.RESULTS / "prepared", A, P, cohort, y, train_pos, test_pos,
                     {**extra_fields(thin), "no_record": {"by_table": by_table, "by_member": by_member},
                      "ext_source_missing": {c: int(Fc[f"{c}_HAS_VALUE"].eq(0).sum()) for c in pr.EXT_SOURCES}})
    (cm.PREPARED / "passive_extra_f32.bin").write_bytes(P_extra.tobytes())
    extra_m = {"rows": int(len(pool)), "ids": [int(i) for i in pool], "seed": cm.EXTRA_SEED,
               "rule": "the first 2,531 applicants of application_train outside the cohort in SHA-256 order (common.draw_extra_pool)",
               "passive": {"width": int(P_extra.shape[1]), "file": "passive_extra_f32.bin", "dtype": "float32 little-endian",
                           "shape": list(P_extra.shape), "sha256": cm.sha256_file(cm.PREPARED / "passive_extra_f32.bin")}}
    cm.write_json(cm.PREPARED / "extra_manifest.json", extra_m)
    shutil.copyfile(cm.PREPARED / "extra_manifest.json", cm.RESULTS / "prepared" / "extra_manifest.json")
    for folder in (cm.PREPARED, cm.RESULTS / "prepared"):
        cm.write_json(folder / "columns.json", {"kept": kept, "dropped": dropped}, indent=0)
        (folder / "README.md").write_text(readme(full, kept, extra_m), encoding="utf-8")

    # the smoke sample: the same encoded rows, positions re-indexed
    at = {int(p): k for k, p in enumerate(smoke)}
    s_train = np.asarray([at[int(p)] for p in train_pos if int(p) in at])
    s_test = np.asarray([at[int(p)] for p in test_pos if int(p) in at])
    s_thin = np.asarray([at[int(p)] for p in thin if int(p) in at], dtype=int)
    sm = write_set(cm.PREPARED_SMOKE, cm.RESULTS / "prepared-smoke", A[smoke], P[smoke], cohort[smoke], y[smoke], s_train, s_test,
                   {**extra_fields(s_thin), "smoke": {"of": "data/mci_loan/prepared", "cohort_pos": [int(p) for p in smoke],
                                                       "rule": "common.smoke_positions: 400 + 400 training, 100 + 100 test rows"}})
    for folder in (cm.PREPARED_SMOKE, cm.RESULTS / "prepared-smoke"):
        cm.write_json(folder / "columns.json", {"kept": kept, "dropped": dropped}, indent=0)
    print(json.dumps({"rows": full["rows"], "bank": sel, "widths": {p: len(v) for p, v in parties.items()},
                      "dropped": [d["name"] for d in dropped], "thin_file": len(thin), "no_record": by_member,
                      "bank_clipped": {c: i["clipped_train"] + i["clipped_test"] for c, i in a_info.items()},
                      "smoke_rows": sm["rows"], "seconds": round(time.time() - t0, 1), "freeRamMB": cm.free_ram_mb()}))


if __name__ == "__main__":
    main()
