import os
import shutil
import argparse

MAPPING = [
    ("train", "hazy", "ITS", "hazy"),
    ("train", "gt",   "ITS", "clear"),
    ("test",  "hazy", "SOTS", "hazy"),
    ("test",  "gt",   "SOTS", "clear"),
]


def link_or_copy(src, dst, copy):
    if os.path.exists(dst) or os.path.islink(dst):
        print(f"  Skipping (already exists): {dst}")
        return
    if copy:
        shutil.copytree(src, dst)
        print(f"  Copied {src} -> {dst}")
    else:
        try:
            os.symlink(os.path.abspath(src), dst, target_is_directory=True)
            print(f"  Linked {src} -> {dst}")
        except (OSError, NotImplementedError) as e:
            print(f"  Symlink failed ({e}); falling back to copy...")
            shutil.copytree(src, dst)
            print(f"  Copied {src} -> {dst}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True,
                     help="Path to the extracted Kaggle RESIDE folder "
                          "(the one containing 'train' and 'test' subfolders)")
    ap.add_argument("--copy", action="store_true",
                     help="Copy files instead of symlinking")
    args = ap.parse_args()

    os.makedirs("dataset/ITS", exist_ok=True)
    os.makedirs("dataset/SOTS", exist_ok=True)

    for split_src, sub_src, split_dst, sub_dst in MAPPING:
        src = os.path.join(args.root, split_src, sub_src)
        dst = os.path.join("dataset", split_dst, sub_dst)

        if not os.path.isdir(src):
            print(f"WARNING: expected folder not found: {src}  (skipping)")
            continue

        print(f"{src}  ->  {dst}")
        link_or_copy(src, dst, args.copy)

    print("\nDone. Verifying counts:")
    for split in ["ITS", "SOTS"]:
        for kind in ["hazy", "clear"]:
            path = os.path.join("dataset", split, kind)
            if os.path.isdir(path):
                n = len(os.listdir(path))
                print(f"  dataset/{split}/{kind}: {n} files")
            else:
                print(f"  dataset/{split}/{kind}: MISSING")


if __name__ == "__main__":
    main()
