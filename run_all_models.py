import os
import sys
import subprocess

for fname in os.listdir(sys.argv[1]):
    if fname.endswith(".pth"):
        tokens = fname.split("_")
        assert tokens[1] == "patch"
        dataset_name = tokens[0]
        num_pts = tokens[2]
        model_path = os.path.join(sys.argv[1], fname)
        data_path = os.path.join("./pclouds", dataset_name, "test", num_pts)
        subprocess.check_call([
            "python",
            "eval_sebastian.py",
            model_path,
            data_path,
            "--points_per_patch",
            num_pts
        ])
