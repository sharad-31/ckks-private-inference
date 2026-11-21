"""Step 10: read the cell images, resize to IMG x IMG grayscale, PATIENT-LEVEL train/val/test split."""
import json, re, numpy as np
from multiprocessing import Pool
from PIL import Image
import config as C

EXT = {".png", ".jpg", ".jpeg"}


def class_files():
    out = []
    for name, label in C.CLASSES.items():
        dirs = [p for p in C.RAW.rglob("*") if p.is_dir() and p.name.lower() == name.lower()]
        dirs = [d for d in dirs if any(f.suffix.lower() in EXT for f in d.iterdir())]
        if not dirs:
            raise FileNotFoundError(f"folder '{name}' with images not found under {C.RAW}")
        out += [(f, label) for f in sorted(dirs[0].iterdir()) if f.suffix.lower() in EXT]
    return out


def load(path):
    im = Image.open(path).convert("L").resize((C.IMG, C.IMG), Image.LANCZOS)
    return np.asarray(im, dtype=np.float32) / 255.0


def group_of(path):
    """patient / slide id = file name up to '_IMG' (e.g. C33P1thinF_IMG_2015..._cell_179.png -> C33P1thinF)"""
    m = re.match(r"^(.+?)_IMG", path.stem, re.I)
    return m.group(1) if m else path.stem


def split_by_group(groups, seed):
    uniq, counts = np.unique(groups, return_counts=True)
    order = np.random.RandomState(seed).permutation(len(uniq))
    n, cum, assign = counts.sum(), 0, {}
    for i in order:
        assign[uniq[i]] = "train" if cum < .70 * n else ("val" if cum < .85 * n else "test")
        cum += counts[i]
    return np.array([assign[g] for g in groups])


def main():
    files = class_files()
    paths = [f for f, _ in files]
    y = np.array([l for _, l in files], dtype=np.int64)
    groups = np.array([group_of(p) for p in paths])
    print(f"{len(paths)} images | {len(np.unique(groups))} patient/slide groups | positives {y.mean():.3f}")
    with Pool(8) as pool:
        X = np.stack(pool.map(load, paths, chunksize=256))
    split = split_by_group(groups, C.SEED)
    for a in ("train", "val", "test"):                       # no patient may appear in two splits
        for b in ("train", "val", "test"):
            if a < b:
                assert not (set(groups[split == a]) & set(groups[split == b])), "patient leakage"

    # preview sheet (raw pixels) for the report
    rng = np.random.RandomState(0)
    tiles = []
    for lab in (0, 1):
        idx = rng.choice(np.where((y == lab) & (split == "train"))[0], 16, replace=False)
        tiles.append(np.concatenate([X[i] for i in idx], axis=1))
    sheet = (np.concatenate(tiles, axis=0) * 255).astype(np.uint8)
    Image.fromarray(sheet).resize((sheet.shape[1] * 3, sheet.shape[0] * 3), Image.NEAREST).save(C.REPORT / f"samples_{C.TAG}.png")

    tr = split == "train"
    mean, std = float(X[tr].mean()), float(X[tr].std() + 1e-6)
    X = np.clip((X - mean) / std, -5, 5).astype(np.float32)   # train-only statistics
    np.savez(C.PROC / f"images_{C.TAG}.npz", X=X, y=y, split=split, group=groups,
             fname=np.array([p.name for p in paths]), mean=mean, std=std)

    info = dict(tag=C.TAG, dataset=C.DATASET, img=C.IMG, n_images=int(len(y)), n_groups=int(len(np.unique(groups))),
                positive_rate=float(y.mean()), mean=mean, std=std,
                split_counts={k: int((split == k).sum()) for k in ("train", "val", "test")},
                split_positive_rate={k: float(y[split == k].mean()) for k in ("train", "val", "test")},
                split_groups={k: int(len(np.unique(groups[split == k]))) for k in ("train", "val", "test")})
    json.dump(info, open(C.REPORT / f"cohort_{C.TAG}.json", "w"), indent=2)
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
