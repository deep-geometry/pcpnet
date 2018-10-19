import os
import point_upsampling_utils as psu

import numpy as np
import torch
from torch.utils.data import Dataset
from dataset import SebastianPatchDataset


class DummySebastianPatchDataset(Dataset):

    def __init__(self, dirname, use_pca=True):
        self.dirname = dirname
        self.shape_names = [os.path.join(dirname, p) for p in os.listdir(dirname)]
        self.use_pca = use_pca

    def __getitem__(self, i):
        patch_path = self.shape_names[i]

        v, _, n = psu.read_obj(patch_path)
        v = torch.FloatTensor(v)
        n = torch.FloatTensor(n)

        if self.use_pca:
            v_mean = v.mean(0)
            v = v - v_mean

            trans, _, _ = torch.svd(torch.t(v))
            v = torch.mm(v, trans)
            cp_new = -v_mean
            cp_new = torch.matmul(cp_new, trans)

            v = v - cp_new
            n = torch.matmul(n, trans)
        else:
            trans = torch.eye(3).float()

        return (v,) + (n,) + (trans,)

    def __len__(self):
        return len(self.shape_names)


def plot_dataset():
    from mayavi import mlab
    patch_dataset = "pclouds/10k_patch/train/512"

    # dataset_no_pca = SebastianPatchDataset(patch_dataset, use_pca=False)
    # dataset_pca = DummySebastianPatchDataset(patch_dataset)

    dataset_pca = SebastianPatchDataset(root = patch_dataset,
                                        shape_list_filename="",
                                        patch_radius=-1,
                                        points_per_patch=1,
                                        patch_features=['normal'])
    pmt = np.random.permutation(len(dataset_pca))
    for i in pmt:
        with_pca = dataset_pca[i]
        # without_pca = dataset_no_pca[i]

        v1, n1, trans1 = with_pca
        mlab.points3d(v1[:, 0], v1[:, 1], v1[:, 2], scale_factor=0.001, color=(0.2, 0.2, 0.8))
        mlab.quiver3d([0], [0], [0], n1[0], n1[1], n1[2], color=(0.2, 0.2, 0.8))

        # v2, n2, trans2 = without_pca
        # mlab.points3d(v2[:, 0], v2[:, 1], v2[:, 2], scale_factor=0.001, color=(0.8, 0.2, 0.2))
        # mlab.quiver3d(v2[:, 0], v2[:, 1], v2[:, 2], n2[:, 0], n2[:, 1], n2[:, 2], color=(0.8, 0.2, 0.2))
        #
        # mlab.points3d([0.0], [0.0], [0.0], scale_factor=0.0025, color=(0.2, 0.8, 0.2))
        # mlab.quiver3d([0.0]*3, [0.0]*3, [0.0]*3, [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
        #               color=(0.1, 0.1, 0.1), scale_factor=0.01)
        mlab.show()


if __name__ == "__main__":
    plot_dataset()