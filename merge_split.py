import glob, os, shutil, sys, collections

val_clips = set(sys.argv[1:])          # clip names to use as validation
imgs = glob.glob("dataset_raw/*/*/images/*.jpg")
clip_of = lambda p: os.path.basename(p).split("__")[0]

counts = collections.Counter(clip_of(p) for p in imgs)
print("Clips found:")
for c, n in sorted(counts.items()):
    print(f"  {c}: {n} images")

if not val_clips:
    print("\nRerun with 1-2 clip names as validation, e.g.:")
    print("python merge_split.py CLIPNAME1 CLIPNAME2")
    sys.exit()

for s in ["train", "val"]:
    os.makedirs(f"dataset/images/{s}", exist_ok=True)
    os.makedirs(f"dataset/labels/{s}", exist_ok=True)

n = collections.Counter()
for p in imgs:
    s = "val" if clip_of(p) in val_clips else "train"
    lbl = p.replace("images", "labels").rsplit(".", 1)[0] + ".txt"
    shutil.copy(p, f"dataset/images/{s}/")
    if os.path.exists(lbl):
        shutil.copy(lbl, f"dataset/labels/{s}/")
    n[s] += 1

with open("dataset/data.yaml", "w") as f:
    f.write("path: /content/dataset\ntrain: images/train\nval: images/val\nnames:\n  0: fish\n")
print(dict(n))