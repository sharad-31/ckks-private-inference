"""Step 13: collect every report into reports/SUMMARY.md and summary.csv"""
import csv, json
import config as C


def load(pat):
    out = []
    for f in sorted(C.REPORT.glob(pat)):
        r = json.load(open(f)); r["_file"] = f.name; out.append(r)
    return out


def fm(x, n=4):
    return "-" if x is None else (f"{x:.{n}f}" if isinstance(x, float) else str(x))


def main():
    cohort = {r["tag"]: r for r in load("cohort_*.json")}
    train = {r["tag"]: r for r in load("train_*.json")}
    ckks, thr, bat = load("ckks_*.json"), load("thr_*.json"), load("batch_*.json")
    md = [f"# Summary - CKKS private inference on images ({C.DATASET})", ""]
    for t, c in cohort.items():
        md += [f"## Data: {t}", "",
               f"- {c['n_images']} images, {c['n_groups']} patient/slide groups, positive rate {c['positive_rate']:.3f}, resized to {c['img']}x{c['img']} grayscale",
               f"- patient-level split (no patient in two splits): {c['split_counts']}, positives {({k: round(v, 3) for k, v in c['split_positive_rate'].items()})}", ""]
        r = train.get(t)
        if r:
            md += ["## Plaintext models (test split)", "", "| model | val AUROC | test AUROC | test accuracy |", "|---|---|---|---|",
                   f"| LR (pixels) | {fm(r['lr_val_auroc'])} | {fm(r['lr_test_auroc'])} | {fm(r['lr_test_acc'])} |",
                   f"| CNN ({C.CHANNELS} ch, {C.KERNEL}x{C.KERNEL}, stride {C.STRIDE}, {r['cnn_params']} params) | {fm(r['cnn_val_auroc'])} | {fm(r['cnn_test_auroc'])} | {fm(r['cnn_test_acc'])} |", ""]
    if ckks:
        md += ["## Encrypted inference (step 12)", "",
               "| model | poly | preset | encrypted on | AUROC plain | AUROC enc | acc plain | acc enc | mean err | max err | signed err | flips | flips (poly-calibrated thr) | s/patient |",
               "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in ckks:
            md.append(f"| {r['model']} | {r['poly']} | {r['preset']} | {r['split']} (n={r['n']}) | {fm(r['auroc_plain'])} | {fm(r['auroc_enc'])} | "
                      f"{fm(r['acc_plain'])} | {fm(r['acc_enc'])} | {r['err_mean']:.2e} | {r['err_max']:.2e} | {r['err_signed_mean']:+.1e} | {r['class_flips']} | {r.get('class_flips_calibrated', r['class_flips'])} | {fm(r['sec_per_patient_wall'], 3)} |")
        md += ["", "Test-split rows:", "", "| model | poly | preset | n | AUROC plain | AUROC enc | acc plain | acc enc | flips |", "|---|---|---|---|---|---|---|---|---|"]
        for r in ckks:
            s = (r.get("per_split") or {}).get("test") or (r if r["split"] == "test" else None)
            if s:
                md.append(f"| {r['model']} | {r['poly']} | {r['preset']} | {s['n']} | {fm(s['auroc_plain'])} | {fm(s['auroc_enc'])} | {fm(s['acc_plain'])} | {fm(s['acc_enc'])} | {s['class_flips']} |")
        md.append("")
    if bat:
        md += ["# Novelty 2 - SIMD batching (step 15)", "",
               "| model | poly | preset | decision (scale pct) | patients | batch | wall s/patient | CPU ms/patient (enc + eval) | upload MB/patient | AUROC enc | flips |",
               "|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in bat:
            md.append(f"| {r['model']} | {r['poly']} | {r['preset']} | {(str(r['decision']) + ' (' + format(r.get('scale_pct', 100), 'g') + ')') if r['decision'] else 'False'} | {r['n']} | {r['batch_size']} | {r['sec_per_patient_wall']:.5f} | "
                      f"{r['cpu_ms_per_patient_eval'] + r['cpu_ms_per_patient_encrypt']:.2f} | {r['upload_MB_per_patient']:.4f} | {fm(r.get('auroc_enc'))} | {r['class_flips']} |")
        md.append("")
    dec = thr + [r for r in bat if r["decision"]]
    if dec:
        md += ["# Novelty 1 - encrypted threshold decision (step 14 and 15 --decision)", "",
               "| model | method | preset | scale pct | sign iters | patients | class flips | acc plain | acc enc | saturated (>=0.9) | Pearson(out, logit) | overflow (|x|>1.7) |",
               "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in dec:
            md.append(f"| {r['model']} | {'batched (15)' if 'batch_size' in r else 'per patient (14)'} | {r['preset']} | {r.get('scale_pct', 100)} | {r['sign_iters']} | {r['n']} | "
                      f"{r['class_flips']} | {fm(r['acc_plain'])} | {fm(r['acc_enc'])} | {fm(r['saturated_fraction'], 3)} | {fm(r['corr_out_vs_logit'], 3)} | {r.get('n_overflow', '-')} |")
        md += ["", "`saturated` = share of patients whose decrypted value is within 0.1 of +-1 (the client learns only the class); "
               "the others are correctly signed but their magnitude still hints at the distance from the threshold. The score is hidden only for saturated patients (the others can be inverted to a score, the polynomial is monotone), so `saturated` is the leakage indicator. Pearson is dominated by a few huge logits and changes a lot with the scale choice alone - do not use it as a leakage measure.", ""]
    (C.REPORT / "SUMMARY.md").write_text("\n".join(md), encoding="utf-8")
    rows = [dict(kind="ckks", **{k: v for k, v in r.items() if not isinstance(v, (dict, list))}) for r in ckks] + \
           [dict(kind="batch", **{k: v for k, v in r.items() if not isinstance(v, (dict, list))}) for r in bat] + \
           [dict(kind="threshold", **{k: v for k, v in r.items() if not isinstance(v, (dict, list))}) for r in thr]
    if rows:
        cols = sorted({k for r in rows for k in r})
        with open(C.REPORT / "summary.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols); w.writeheader(); w.writerows(rows)
    print("\n".join(md)); print("saved", C.REPORT / "SUMMARY.md")


if __name__ == "__main__":
    main()
