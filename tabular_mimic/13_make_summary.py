"""Step 13: collect every report in reports/ into SUMMARY.md + summary.csv
(all datasets / grids / presets that were run)."""
import json, csv
import config as C


def load(pattern):
    return [json.load(open(f)) for f in sorted(C.REPORT.glob(pattern))]


def main():
    cohorts = {r["tag"]: r for r in load("cohort_*.json")}
    trains  = {r["tag"]: r for r in load("train_*.json")}
    ckks    = load("ckks_*.json")
    f = lambda x, n=4: "-" if x is None else (f"{x:.{n}f}" if isinstance(x, float) else str(x))
    tags = sorted(set(cohorts) | set(trains) | {r["tag"] for r in ckks})

    rows = []
    for t in tags:
        c, tr = cohorts.get(t, {}), trains.get(t, {})
        base = dict(tag=t, dataset=c.get("dataset", tr.get("dataset")),
                    grid="x".join(map(str, c.get("grid", tr.get("grid", [])))),
                    target=c.get("target", tr.get("target")),
                    n_stays=c.get("n_stays"), positive_rate=c.get("positive_rate"),
                    lr_test_auroc=tr.get("lr_test_auroc"), cnn_test_auroc=tr.get("cnn_test_auroc"),
                    cnn_test_auprc=tr.get("cnn_test_auprc"))
        runs = [r for r in ckks if r["tag"] == t] or [{}]
        for r in runs:
            rows.append({**base, "preset": r.get("preset"), "split": r.get("split", "test"), "n_enc": r.get("n"),
                         "enc_auroc": r.get("auroc_enc"), "logit_err_max": r.get("logit_err_max"),
                         "class_flips": r.get("class_flips"),
                         "sec_per_patient": r.get("sec_per_patient_wall")})

    cols = list(rows[0].keys()) if rows else []
    with open(C.REPORT / "summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols); w.writeheader(); w.writerows(rows)

    md = ["# Summary - CKKS private inference with hybrid CNN (time-series grid + static LR branch)", ""]
    md += ["## All runs", "",
           "| dataset | grid | target | stays | pos.rate | static-LR AUROC | CNN AUROC | preset | encrypted on | enc AUROC | max logit err | flips | s/patient |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append("| " + " | ".join([
            f(r["dataset"]), f(r["grid"]), f(r["target"]), f(r["n_stays"]), f(r["positive_rate"], 3),
            f(r["lr_test_auroc"]), f(r["cnn_test_auroc"]), f(r["preset"]),
            (f"{r['split']} (n={r['n_enc']})" if r["n_enc"] else "-"), f(r["enc_auroc"]),
            f"{r['logit_err_max']:.2e}" if r["logit_err_max"] is not None else "-",
            f(r["class_flips"]), f(r["sec_per_patient"], 3)]) + " |")
    md += ["", "Static-LR and CNN AUROC columns are on the held-out test split (patient-level split). `encrypted on` = which patients were encrypted; `all` is the whole cohort (train + val + test), so its AUROC is optimistic - see the per-split lines below for test-only numbers. "
           "`flips` = encrypted prediction differs from plaintext prediction at the val-tuned threshold.", ""]
    for t in tags:
        c, tr = cohorts.get(t), trains.get(t)
        md += [f"## {t}", ""]
        if c:
            md += [f"- cohort: {c['n_stays']} stays, positive rate {c['positive_rate']:.3f}, "
                   f"split {c['split_counts']}",
                   f"- vitals in grid ({len(c['vitals_in_grid'])}): {', '.join(c['vitals_in_grid'])}",
                   "- stays with at least one observation: " +
                   ", ".join(f"{k} {v*100:.0f}%" for k, v in c["vital_coverage"].items())]
        if tr:
            md += [f"- static-only LR: test AUROC {tr['lr_test_auroc']:.4f}",
                   f"- CNN: val AUROC {tr['cnn_val_auroc']:.4f}, test AUROC {tr['cnn_test_auroc']:.4f}, "
                   f"test AUPRC {tr['cnn_test_auprc']:.4f}, threshold {tr['threshold']:.2f}"]
        for r in ckks:
            if r["tag"] == t:
                md.append(f"- CKKS {r['preset']} on {r.get('split','test')}: n={r['n']}, enc AUROC {r['auroc_enc']:.4f}, "
                          f"max logit err {r['logit_err_max']:.2e}, flips {r['class_flips']}, "
                          f"{(r['sec_per_patient_wall'] or 0):.2f} s/patient (wall, parallel)")
                for k, v in (r.get("per_split") or {}).items():
                    md.append(f"    - {k}: n={v['n']}, plain AUROC {v['auroc_plain']:.4f}, "
                              f"enc AUROC {v['auroc_enc']:.4f}, flips {v['class_flips']}, "
                              f"acc plain/enc {v['acc_plain']:.4f}/{v['acc_enc']:.4f}")
        md.append("")
    (C.REPORT / "SUMMARY.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    print("saved", C.REPORT / "SUMMARY.md", "and summary.csv")


if __name__ == "__main__":
    main()
