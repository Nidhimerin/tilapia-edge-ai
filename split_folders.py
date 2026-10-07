import glob, os, shutil

imgs = sorted(glob.glob("frames_train/*.jpg"))
people = ["krishna", "nileena", "nidhi"]
for p in people:
    os.makedirs(f"batches/{p}", exist_ok=True)
for i, img in enumerate(imgs):
    dest = f"batches/{people[i % 3]}"
    shutil.copy(img, dest)
    txt = img.replace(".jpg", ".txt")
    if os.path.exists(txt):
        shutil.copy(txt, dest)
for p in people:
    shutil.copy("frames_train/data.yaml", f"batches/{p}/data.yaml")
    shutil.make_archive(f"batches/{p}", "zip", f"batches/{p}")
print(len(imgs), "images split")