import torch
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("data_filename", type=str)
parser.add_argument("--num-buckets", type=int, default=-1)
parser.add_argument("--title", type=str, default="Title")
args = parser.parse_args()

print("Loading data...")
data = torch.load(args.data_filename)
print("Computing losses...")
losses = [float(l['one_minus_cos_loss']) for l in data]

print("There are %d losses" % len(losses))
print("Plotting histogram...")
if args.num_buckets > 0:
    plt.hist(losses, args.num_buckets)
else:
    plt.hist(losses)

plt.title(args.title)
plt.show()
