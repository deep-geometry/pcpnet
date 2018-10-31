from __future__ import print_function

import argparse
import os
import random
import torch
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import utils
from dataset import SebastianPatchDataset
from pcpnet import PCPNet, MSPCPNet
import numpy as np


def parse_arguments():
    parser = argparse.ArgumentParser()

    # naming / file handling
    parser.add_argument('model_filename', type=str, help='training run name')
    parser.add_argument('datadir', type=str, help='directory of dataset')

    # training parameterse
    parser.add_argument('--batchSize', type=int, default=64, help='input batch size')

    # unused
    parser.add_argument('--patch_radius', type=float, default=[0.05], nargs='+', help='patch radius in multiples of the shape\'s bounding box diagonal, multiple values for multi-scale.')
    parser.add_argument('--patch_center', type=str, default='point', help='center patch at...\n'
                        'point: center point\n'
                        'mean: patch mean')
    parser.add_argument('--patch_point_count_std', type=float, default=0, help='standard deviation of the number of points in a patch')
    parser.add_argument('--patches_per_shape', type=int, default=1, help='number of patches sampled from each shape in an epoch')
    parser.add_argument('--cache_capacity', type=int, default=100, help='Max. number of dataset elements (usually shapes) to hold in the cache at the same time.')
    parser.add_argument('--identical_epochs', type=int, default=False, help='use same patches in each epoch, mainly for debugging')

    parser.add_argument('--workers', type=int, default=1, help='number of data loading workers - 0 means same thread as main execution')
    parser.add_argument('--seed', type=int, default=3627473, help='manual seed')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='gradient descent momentum')
    parser.add_argument('--use_pca', type=int, default=False, help='Give both inputs and ground truth in local PCA coordinate frame')
    parser.add_argument('--normal_loss', type=str, default='ms_oneminuscos', help='Normal loss type:\n'
                        'ms_euclidean: mean square euclidean distance\n'
                        'ms_oneminuscos: mean square 1-cos(angle error)')

    # model hyperparameters
    parser.add_argument('--outputs', type=str, nargs='+', default=['unoriented_normals'], help='outputs of the network, a list with elements of:\n'
                        'unoriented_normals: unoriented (flip-invariant) point normals\n'
                        'oriented_normals: oriented point normals\n'
                        'max_curvature: maximum curvature\n'
                        'min_curvature: mininum curvature')
    parser.add_argument('--use_point_stn', type=int, default=True, help='use point spatial transformer')
    parser.add_argument('--use_feat_stn', type=int, default=True, help='use feature spatial transformer')
    parser.add_argument('--sym_op', type=str, default='max', help='symmetry operation')
    parser.add_argument('--point_tuple', type=int, default=1, help='use n-tuples of points as input instead of single points')
    parser.add_argument('--points_per_patch', type=int, default=500, help='max. number of points per patch')

    return parser.parse_args()


def eval_pcpnet(opt):
    model_filename = opt.model_filename
    losses_filename = opt.model_filename + ".eval_data"

    target_features = []
    output_target_ind = []
    output_pred_ind = []
    output_loss_weight = []
    pred_dim = 0

    for o in opt.outputs:
        if o == 'oriented_normals' or o == 'unoriented_normals':
            if 'normal' not in target_features:
                target_features.append('normal')

            output_target_ind.append(target_features.index('normal'))
            output_pred_ind.append(pred_dim)
            output_loss_weight.append(1.0)
            pred_dim += 3
        else:
            raise ValueError('Unknown output: %s' % (o))

    if pred_dim <= 0:
        raise ValueError('Prediction is empty for the given outputs.')

    # create model
    if len(opt.patch_radius) == 1:
        pcpnet = PCPNet(
            num_points=opt.points_per_patch,
            output_dim=pred_dim,
            use_point_stn=opt.use_point_stn,
            use_feat_stn=opt.use_feat_stn,
            sym_op=opt.sym_op,
            point_tuple=opt.point_tuple)
    else:
        assert False, "Sebastian only supports patch_radius size 1"

    if os.path.exists(model_filename):
        pcpnet.load_state_dict(torch.load(model_filename))
    else:
        raise ValueError("No model to load")

    if opt.seed < 0:
        opt.seed = random.randint(1, 10000)

    print("Random Seed: %d" % (opt.seed))
    random.seed(opt.seed)
    torch.manual_seed(opt.seed)

    # create train and test dataset loaders
    train_dataset = SebastianPatchDataset(
        root=opt.datadir,
        shape_list_filename="",
        patch_radius=opt.patch_radius,
        points_per_patch=opt.points_per_patch,
        patch_features=target_features,
        point_count_std=opt.patch_point_count_std,
        seed=opt.seed,
        identical_epochs=opt.identical_epochs,
        use_pca=opt.use_pca,
        center=opt.patch_center,
        point_tuple=opt.point_tuple,
        cache_capacity=opt.cache_capacity,
        output_eval_data=True)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=False,
        batch_size=opt.batchSize,
        num_workers=int(opt.workers))

    # keep the exact training shape names for later reference
    opt.train_shapes = train_dataset.shape_names

    print('evaluation dataset: %d patches (in %d batches)' %
          (len(train_dataset), len(train_dataloader)))

    pcpnet.cuda()

    train_enum = enumerate(train_dataloader, 0)

    prediction_data = []

    for train_batchind, data in train_enum:

        # set to evaluation mode
        pcpnet.eval()

        # get testset batch, convert to variables and upload to GPU
        # volatile means that autograd is turned off for everything that depends on the volatile variable
        # since we dont need autograd for inference (only for training)
        points = data[0]
        target = data[1:-3]
        filenames = data[-2]
        point_indexes = data[-1]

        points = Variable(points, volatile=True)
        points = points.transpose(2, 1)
        points = points.cuda()

        target = tuple(Variable(t, volatile=True) for t in target)
        target = tuple(t.cuda() for t in target)

        # forward pass
        pred, trans, _, _ = pcpnet(points)

        losses = compute_loss(
            pred=pred, target=target,
            outputs=opt.outputs,
            output_pred_ind=output_pred_ind,
            output_target_ind=output_target_ind,
            output_loss_weight=output_loss_weight,
            patch_rot=trans if opt.use_point_stn else None,
            normal_loss=opt.normal_loss)

        o_pred = torch.bmm(pred.unsqueeze(1), trans.transpose(2, 1)).squeeze(1)

        for i in range(o_pred.shape[0]):
            target_i = target[0][i, :].data.cpu().numpy()
            predicted_i = o_pred[i, :].data.cpu().numpy()
            target_i = target_i / np.linalg.norm(target_i)
            predicted_i = predicted_i / np.linalg.norm(predicted_i)

            my_loss = float(1.0 - np.abs(np.dot(target_i, predicted_i)))**2
            their_loss = float(losses[i].data.cpu())

            # print("filename: %s" % filenames[i])
            # print("  expected: %s" % str(target_i))
            # print("  predicted %s" % str(predicted_i))
            # print("  computed_loss1 %f" % their_loss)
            # print("  computed_loss2 %f" % my_loss)
            infodict = {
                'filename': filenames[i],
                'expected_normal': target_i,
                'predicted_normal': predicted_i,
                'one_minus_cos_loss': float(losses[i].data.cpu().numpy()),
                'ctr_idx': point_indexes[i],
            }
            prediction_data.append(infodict)

    torch.save(prediction_data, losses_filename)


def compute_loss(pred, target, outputs, output_pred_ind, output_target_ind, output_loss_weight, patch_rot, normal_loss):

    assert len(list(enumerate(outputs))) == 1, "bad number of outputs"

    losses = []
    for oi, o in enumerate(outputs):
        if o == 'unoriented_normals':
            o_pred = pred[:, output_pred_ind[oi]:output_pred_ind[oi]+3]
            o_target = target[output_target_ind[oi]]

            if patch_rot is not None:
                # transform predictions with inverse transform
                # since we know the transform to be a rotation (QSTN), the transpose is the inverse
                o_pred = torch.bmm(o_pred.unsqueeze(1), patch_rot.transpose(2, 1)).squeeze(1)

            if normal_loss == 'ms_oneminuscos':
                l = (1-torch.abs(utils.cos_angle(o_pred, o_target))).pow(2) * output_loss_weight[oi]
                ll = [l[i] for i in range(l.shape[0])]
                losses.extend(ll)
            else:
                raise ValueError('Unsupported output type: %s' % (o))
        else:
            raise ValueError('Unsupported output type: %s' % (o))

    return losses


if __name__ == '__main__':
    train_opt = parse_arguments()
    eval_pcpnet(train_opt)
    
