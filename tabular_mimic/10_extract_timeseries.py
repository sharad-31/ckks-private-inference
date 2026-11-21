"""Step 10: cohort + 12 x N_VITALS time-series grid + built-in static features.
Works for MIMIC-III and MIMIC-IV (config.DATASET or --dataset).
"""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import duckdb

# Add project root to sys.path
ROOT_REPO = Path(__file__).resolve().parent.parent
if str(ROOT_REPO) not in sys.path:
    sys.path.insert(0, str(ROOT_REPO))

import config as C


def read_table(name):
    df = pd.read_csv(C.find_table(name), low_memory=False)
    df.columns = df.columns.str.lower()
    return df


def build_cohort():
    stay = read_table("icustays")
    adm  = read_table("admissions")[["hadm_id", "hospital_expire_flag", "admission_type"]]
    pat  = read_table("patients")
    if C.DATASET == "mimic3":
        stay = stay.rename(columns={"icustay_id": "stay_id"})
    stay["intime"] = pd.to_datetime(stay.intime)
    if C.DATASET == "mimic4":
        pat = pat[["subject_id", "anchor_age", "gender"]]
        df = stay.merge(pat, on="subject_id").merge(adm, on="hadm_id", how="left")
        df["age"] = df.anchor_age
    else:                                    # mimic3: age = year(intime) - year(dob); >89 is date-shifted
        pat = pat[["subject_id", "dob", "gender"]]
        df = stay.merge(pat, on="subject_id").merge(adm, on="hadm_id", how="left")
        df["age"] = (df.intime.dt.year - pd.to_datetime(df.dob, errors="coerce").dt.year).clip(upper=90)
    df = df[(df.age >= 18) & (df.los >= C.WINDOW_H / 24)]
    df = df.sort_values(["subject_id", "intime"]).drop_duplicates("subject_id")
    df["los7"] = (df.los > 7).astype(int)
    df["mortality"] = df.hospital_expire_flag.fillna(0).astype(int)
    df["y"] = df[C.TARGET]

    ids = np.sort(df.subject_id.unique())            # deterministic split by patient
    np.random.RandomState(C.SEED).shuffle(ids)
    n = len(ids)
    split = {s: ("train" if i < .70 * n else ("val" if i < .85 * n else "test"))
             for i, s in enumerate(ids)}
    df["split"] = df.subject_id.map(split)
    return df.reset_index(drop=True)


def extract_bins(cohort):
    item2v = {i: v for v, ids in C.VITALS_ALL.items() for i in ids}   # all 10 vitals once
    con = duckdb.connect()
    con.register("cohort", cohort[["stay_id", "intime"]])
    path = C.find_table("chartevents").as_posix()
    # actual column names in the file (any case) -> quoted lookup
    hdr = con.execute(f"DESCRIBE SELECT * FROM read_csv('{path}', header=true, "
                      f"all_varchar=true, sample_size=1000)").df().column_name
    col = {h.lower(): '"' + h + '"' for h in hdr}
    idcol = col["icustay_id" if C.DATASET == "mimic3" else "stay_id"]
    err = (f"TRY_CAST({col['error']} AS INTEGER)" if "error" in col else "CAST(NULL AS INTEGER)")
    ids = ",".join(map(str, item2v))
    q = f"""
    SELECT stay_id, itemid, bin, avg(v) AS v FROM (
        SELECT k.stay_id, e.itemid, e.valuenum AS v,
               CAST(floor(date_diff('second', k.intime, e.charttime) / 3600.0
                          / {C.BIN_HOURS}) AS INTEGER) AS bin
        FROM (SELECT TRY_CAST({idcol} AS BIGINT) AS stay_id,
                     TRY_CAST({col['itemid']} AS INTEGER) AS itemid,
                     TRY_CAST({col['charttime']} AS TIMESTAMP) AS charttime,
                     TRY_CAST({col['valuenum']} AS DOUBLE) AS valuenum,
                     {err} AS err
              FROM read_csv('{path}', header=true, all_varchar=true,
                            ignore_errors=true)) e
        JOIN cohort k ON k.stay_id = e.stay_id
        WHERE e.itemid IN ({ids}) AND e.valuenum IS NOT NULL
          AND (e.err IS NULL OR e.err = 0))
    WHERE bin >= 0 AND bin < {C.N_BINS}
    GROUP BY stay_id, itemid, bin
    """
    d = con.execute(q).df()
    d["vital"] = d.itemid.map(item2v)
    d.loc[d.itemid.isin(C.TEMP_F_ITEMS), "v"] = (d.v - 32) * 5 / 9
    d.loc[(d.vital == "fio2") & (d.v <= 1.0), "v"] *= 100
    lo = d.vital.map(lambda x: C.RANGES[x][0]); hi = d.vital.map(lambda x: C.RANGES[x][1])
    d = d[(d.v >= lo) & (d.v <= hi)]
    return d.groupby(["stay_id", "vital", "bin"], as_index=False).v.mean()


def build_static(cohort, d):
    """built-in static features: age, sex, admission type, first care unit, first-24h vital means"""
    s = pd.DataFrame({"stay_id": cohort.stay_id, "age": cohort.age.astype(float),
                      "male": (cohort.gender == "M").astype(float)})
    for col in ("admission_type", "first_careunit"):
        if col in cohort.columns:
            oh = pd.get_dummies(cohort[col].fillna("UNK"), prefix=col, dtype=float)
            s = pd.concat([s, oh], axis=1)
    e = d[(d.bin < 6) & d.vital.isin(C.STATIC_VITALS)]
    m = e.groupby(["stay_id", "vital"]).v.mean().unstack()
    m.columns = [f"{c}_24h_mean" for c in m.columns]
    s = s.merge(m.reset_index(), on="stay_id", how="left")
    return s


def main():
    ap = argparse.ArgumentParser(description="Extract cohort and 12xN time-series grid from MIMIC")
    ap.add_argument("--dataset", choices=["mimic3", "mimic4"], default=C.DATASET, help="MIMIC dataset version")
    ap.add_argument("--vitals", type=int, choices=[5, 10], default=C.N_VITALS, help="Number of vitals in grid (10 -> 12x10, 5 -> 12x5)")
    ap.add_argument("--target", choices=["los7", "mortality"], default=C.TARGET, help="Prediction target")
    args = ap.parse_args()

    # Apply overrides if provided
    C.DATASET = args.dataset
    C.N_VITALS = args.vitals
    C.TARGET = args.target
    C.GRID = (C.N_BINS, C.N_VITALS)
    C.KERNEL_W = C.N_VITALS
    C.VITALS_ALL = {v: ids[C.DATASET] for v, ids in C.VITAL_IDS.items()}
    C.VITALS = dict(list(C.VITALS_ALL.items())[:C.N_VITALS])
    C.TAG = f"{C.DATASET}_{C.N_BINS}x{C.N_VITALS}_{C.TARGET}"

    cohort = build_cohort()
    print(f"[{C.DATASET}] cohort: {len(cohort)} stays | {C.TARGET} rate {cohort.y.mean():.3f} "
          f"| grid {C.N_BINS}x{C.N_VITALS}")
    d_all = extract_bins(cohort)
    d = d_all[d_all.vital.isin(C.VITALS)]

    sid = {s: i for i, s in enumerate(cohort.stay_id)}
    vid = {v: i for i, v in enumerate(C.VITALS)}
    X = np.full((len(cohort), C.N_BINS, C.N_VITALS), np.nan, dtype=np.float32)
    X[d.stay_id.map(sid).values, d.bin.values, d.vital.map(vid).values] = d.v.values

    tr = (cohort.split == "train").values
    mean = np.nanmean(X[tr], axis=(0, 1)); std = np.nanstd(X[tr], axis=(0, 1)) + 1e-6
    X = (X - mean) / std                              # train-only stats
    for t in range(1, C.N_BINS):                      # forward fill in time
        m = np.isnan(X[:, t]); X[:, t][m] = X[:, t - 1][m]
    X = np.clip(np.nan_to_num(X, nan=0.0), -5, 5)     # leftover = train mean

    cover = {v: float(d_all[d_all.vital == v].stay_id.nunique() / len(cohort)) for v in C.VITALS_ALL}
    for v in C.VITALS_ALL:
        print(f"  {v:11s} stays with >=1 obs: {cover[v]*100:5.1f}%" + ("" if v in C.VITALS else "  (not in grid)"))

    np.savez(C.PROC / f"ts_{C.TAG}.npz", X=X, y=cohort.y.values, stay_id=cohort.stay_id.values,
             split=cohort.split.values, vitals=np.array(list(C.VITALS)), mean=mean, std=std)
    if not C.STATIC_CSV.exists() or C.STATIC_CSV.name.startswith("static_builtin"):
        build_static(cohort, d_all).to_csv(C.PROC / f"static_builtin_{C.DATASET}.csv", index=False)
    cohort[["subject_id", "hadm_id", "stay_id", "age", "los", "los7", "mortality", "split"]].to_csv(
        C.PROC / f"cohort_{C.DATASET}.csv", index=False)

    info = dict(tag=C.TAG, dataset=C.DATASET, grid=list(C.GRID), target=C.TARGET,
                n_stays=int(len(cohort)), positive_rate=float(cohort.y.mean()),
                split_counts={k: int((cohort.split == k).sum()) for k in ("train", "val", "test")},
                vital_coverage=cover, vitals_in_grid=list(C.VITALS))
    json.dump(info, open(C.REPORT / f"cohort_{C.TAG}.json", "w"), indent=2)
    print("saved ts grid", X.shape)


if __name__ == "__main__":
    main()
