from __future__ import print_function

import argparse
import os
import sys
import random
import math
import shutil
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
from torch.autograd import Variable
from tensorboardX import SummaryWriter # https://github.com/lanpa/tensorboard-pytorch
import utils
from dataset import SebastianPatchDataset, RandomPointcloudPatchSampler, SequentialShapeRandomPointcloudPatchSampler
from pcpnet import PCPNet, MSPCPNet
import json


def parse_arguments():
    parser = argparse.ArgumentParser()

    # naming / file handling
    parser.add_argument('--name', type=str, default='my_single_scale_normal', help='training run name')
    parser.add_argument('--desc', type=str, default='My training run for single-scale normal estimation.', help='description')
    parser.add_argument('--traindir', type=str, default='./pclouds', help='input folder (point clouds)')
    parser.add_argument('--testdir', type=str, default='./pclouds', help='input folder (point clouds)')
    parser.add_argument('--outdir', type=str, default='./models', help='output folder (trained models)')
    parser.add_argument('--logdir', type=str, default='./logs', help='training log folder')
    parser.add_argument('--trainset', type=str, default='trainingset_whitenoise.txt', help='training set file name')
    parser.add_argument('--testset', type=str, default='validationset_whitenoise.txt', help='test set file name')
    parser.add_argument('--saveinterval', type=int, default='10', help='save model each n epochs')
    parser.add_argument('--refine', type=str, default='', help='refine model at this path')

    # training parameters
    parser.add_argument('--nepoch', type=int, default=2000, help='number of epochs to train for')
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

    parser.add_argument('--training_order', type=str, default='random', help='order in which the training patches are presented:\n'
                        'random: fully random over the entire dataset (the set of all patches is permuted)\n'
                        'random_shape_consecutive: random over the entire dataset, but patches of a shape remain consecutive (shapes and patches inside a shape are permuted)')
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

    # colored console output
    green = lambda x: '\033[92m' + x + '\033[0m'
    blue = lambda x: '\033[94m' + x + '\033[0m'

    log_dirname = os.path.join(opt.logdir, opt.name)
    params_filename = os.path.join(opt.outdir, '%s_params.pth' % (opt.name))
    model_filename = os.path.join(opt.outdir, '%s_model.pth' % (opt.name))
    desc_filename = os.path.join(opt.outdir, '%s_description.txt' % (opt.name))
    params_json_filename = os.path.join(opt.outdir, "%s_params.json" % opt.name)
    # with open(params_json_filename, 'w') as f:
    #     json.dump(vars(opt), f)

    # if os.path.exists(log_dirname) or os.path.exists(model_filename):
    #     response = input('A training run named "%s" already exists, overwrite? (y/n) ' % (opt.name))
    #     if response == 'y':
    #         if os.path.exists(log_dirname):
    #             shutil.rmtree(os.path.join(opt.logdir, opt.name))
    #     else:
    #         sys.exit()

    # get indices in targets and predictions corresponding to each output
    target_features = []
    output_target_ind = []
    output_pred_ind = []
    output_loss_weight = []
    pred_dim = 0
    for o in opt.outputs:
        if o == 'unoriented_normals' or o == 'oriented_normals':
            if 'normal' not in target_features:
                target_features.append('normal')

            output_target_ind.append(target_features.index('normal'))
            output_pred_ind.append(pred_dim)
            output_loss_weight.append(1.0)
            pred_dim += 3
        elif o == 'max_curvature' or o == 'min_curvature':
            if o not in target_features:
                target_features.append(o)

            output_target_ind.append(target_features.index(o))
            output_pred_ind.append(pred_dim)
            if o == 'max_curvature':
                output_loss_weight.append(0.7)
            else:
                output_loss_weight.append(0.3)
            pred_dim += 1
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
        pcpnet = MSPCPNet(
            num_scales=len(opt.patch_radius),
            num_points=opt.points_per_patch,
            output_dim=pred_dim,
            use_point_stn=opt.use_point_stn,
            use_feat_stn=opt.use_feat_stn,
            sym_op=opt.sym_op,
            point_tuple=opt.point_tuple)
        raise ValueError("Sebastian does not support MSPCPNet")

    if os.path.exists(log_dirname) or os.path.exists(model_filename):
        pcpnet.load_state_dict(torch.load(os.path.join(log_dirname, model_filename)))
    else:
        raise ValueError("No Model to load")

    if opt.seed < 0:
        opt.seed = random.randint(1, 10000)

    print("Random Seed: %d" % (opt.seed))
    random.seed(opt.seed)
    torch.manual_seed(opt.seed)

    # create train and test dataset loaders
    train_dataset = SebastianPatchDataset(
        root=opt.traindir,
        shape_list_filename=opt.trainset,
        patch_radius=opt.patch_radius,
        points_per_patch=opt.points_per_patch,
        patch_features=target_features,
        point_count_std=opt.patch_point_count_std,
        seed=opt.seed,
        identical_epochs=opt.identical_epochs,
        use_pca=opt.use_pca,
        center=opt.patch_center,
        point_tuple=opt.point_tuple,
        cache_capacity=opt.cache_capacity)

    shuffle = False
    if opt.training_order == 'random':
        shuffle = True
    elif opt.training_order == 'random_shape_consecutive':
        shuffle = False
    else:
        raise ValueError('Unknown training order: %s' % (opt.training_order))

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=shuffle,
        # sampler=train_datasampler,
        batch_size=opt.batchSize,
        num_workers=int(opt.workers))

    test_dataset = SebastianPatchDataset(
        root=opt.testdir,
        shape_list_filename=opt.testset,
        patch_radius=opt.patch_radius,
        points_per_patch=opt.points_per_patch,
        patch_features=target_features,
        point_count_std=opt.patch_point_count_std,
        seed=opt.seed,
        identical_epochs=opt.identical_epochs,
        use_pca=opt.use_pca,
        center=opt.patch_center,
        point_tuple=opt.point_tuple,
        cache_capacity=opt.cache_capacity)

    shuffle = False
    if opt.training_order == 'random':
        shuffle = True
    elif opt.training_order == 'random_shape_consecutive':
        shuffle = False
    else:
        raise ValueError('Unknown training order: %s' % (opt.training_order))

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=shuffle,
        # sampler=test_datasampler,
        batch_size=opt.batchSize,
        num_workers=int(opt.workers))

    # keep the exact training shape names for later reference
    opt.train_shapes = train_dataset.shape_names
    opt.test_shapes = test_dataset.shape_names

    print('training set: %d patches (in %d batches) - test set: %d patches (in %d batches)' %
          (len(train_dataset), len(train_dataloader), len(train_dataset), len(test_dataloader)))

    try:
        os.makedirs(opt.outdir)
    except OSError:
        pass

    optimizer = optim.SGD(pcpnet.parameters(), lr=opt.lr, momentum=opt.momentum)
    pcpnet.cuda()

    train_enum = enumerate(train_dataloader, 0)

    for train_batchind, data in train_enum:

        # set to evaluation mode
        pcpnet.eval()

        # get testset batch, convert to variables and upload to GPU
        # volatile means that autograd is turned off for everything that depends on the volatile variable
        # since we dont need autograd for inference (only for training)
        points = data[0]
        target = data[1:-1]

        points = Variable(points, volatile=True)
        points = points.transpose(2, 1)
        points = points.cuda()

        target = tuple(Variable(t, volatile=True) for t in target)
        target = tuple(t.cuda() for t in target)

        # forward pass
        pred, trans, _, _ = pcpnet(points)

        loss = compute_loss(
            pred=pred, target=target,
            outputs=opt.outputs,
            output_pred_ind=output_pred_ind,
            output_target_ind=output_target_ind,
            output_loss_weight=output_loss_weight,
            patch_rot=trans if opt.use_point_stn else None,
            normal_loss=opt.normal_loss)

        print(loss)


def compute_loss(pred, target, outputs, output_pred_ind, output_target_ind, output_loss_weight, patch_rot, normal_loss):

    loss = 0

    assert len(enumerate(outputs)) == 1, "bad number of outputs"
    print(output_loss_weight)

    for oi, o in enumerate(outputs):
        if o == 'unoriented_normals' or o == 'oriented_normals':
            o_pred = pred[:, output_pred_ind[oi]:output_pred_ind[oi]+3]
            o_target = target[output_target_ind[oi]]

            if patch_rot is not None:
                # transform predictions with inverse transform
                # since we know the transform to be a rotation (QSTN), the transpose is the inverse
                o_pred = torch.bmm(o_pred.unsqueeze(1), patch_rot.transpose(2, 1)).squeeze(1)

            if o == 'unoriented_normals':
                if normal_loss == 'ms_euclidean':
                    loss += torch.min((o_pred-o_target).pow(2).sum(1), (o_pred+o_target).pow(2).sum(1)).mean() * output_loss_weight[oi]
                elif normal_loss == 'ms_oneminuscos':
                    return (1-torch.abs(utils.cos_angle(o_pred, o_target))).pow(2) * output_loss_weight[oi]
                else:
                    raise ValueError('Unsupported loss type: %s' % (normal_loss))
            elif o == 'oriented_normals':
                if normal_loss == 'ms_euclidean':
                    loss += (o_pred-o_target).pow(2).sum(1).mean() * output_loss_weight[oi]
                elif normal_loss == 'ms_oneminuscos':
                    loss += (1-utils.cos_angle(o_pred, o_target)).pow(2).mean() * output_loss_weight[oi]
                else:
                    raise ValueError('Unsupported loss type: %s' % (normal_loss))
            else:
                raise ValueError('Unsupported output type: %s' % (o))

        elif o == 'max_curvature' or o == 'min_curvature':
            o_pred = pred[:, output_pred_ind[oi]:output_pred_ind[oi]+1]
            o_target = target[output_target_ind[oi]]

            # Rectified mse loss: mean square of (pred - gt) / max(1, |gt|)
            normalized_diff = (o_pred - o_target) / torch.clamp(torch.abs(o_target), min=1)
            loss += normalized_diff.pow(2).mean() * output_loss_weight[oi]

        else:
            raise ValueError('Unsupported output type: %s' % (o))

    return loss


if __name__ == '__main__':
    train_opt = parse_arguments()
    eval_pcpnet(train_opt)
