print# -----------------------------------------------------
# Copyright (c) Shanghai Jiao Tong University. All rights reserved.
# Written by Jiefeng Li (jeff.lee.sjtu@gmail.com)
# -----------------------------------------------------
# HybrIK + BTS
from collections import namedtuple

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

from .builder import SPPE
from .layers.Resnet import ResNet
from .layers.smpl.SMPL import SMPL_layer
from .bts import encoder as bts_encoder
from .bts import bts as bts_decoder
from .bts import DepthParams
from .layers.fusion_modules import PoseDepthFusion2D

ModelOutput = namedtuple(
    typename='ModelOutput',
    field_names=['pred_shape', 'pred_theta_mats', 'pred_phi', 'pred_delta_shape', 'pred_leaf',
                 'pred_uvd_jts', 'pred_xyz_jts_24', 'pred_xyz_jts_24_struct',
                 'pred_xyz_jts_17', 'pred_vertices', 'maxvals']
)
ModelOutput.__new__.__defaults__ = (None,) * len(ModelOutput._fields)


def norm_heatmap(norm_type, heatmap):
    # Input tensor shape: [N,C,...]
    shape = heatmap.shape
    if norm_type == 'softmax':
        heatmap = heatmap.reshape(*shape[:2], -1)
        # global soft max
        heatmap = F.softmax(heatmap, 2)
        return heatmap.reshape(*shape)
    else:
        raise NotImplementedError

# Hybrik + BTS
@SPPE.register_module
class HybrikBTS(nn.Module):
    def __init__(self, norm_layer=nn.BatchNorm2d, **kwargs):
        super(HybrikBTS, self).__init__()
        self.deconv_dim = kwargs['NUM_DECONV_FILTERS']
        self._norm_layer = norm_layer
        self.num_joints = 24
        self.norm_type = kwargs['POST']['NORM_TYPE']
        self.depth_dim = kwargs['EXTRA']['DEPTH_DIM']
        self.height_dim = kwargs['HEATMAP_SIZE'][0]
        self.width_dim = kwargs['HEATMAP_SIZE'][1]
        self.smpl_dtype = torch.float32

        backbone = ResNet

        self.preact = backbone(f"resnet{kwargs['NUM_LAYERS']}")

        # Imagenet pretrain model
        import torchvision.models as tm
        if kwargs['NUM_LAYERS'] == 101:
            ''' Load pretrained model '''
            x = tm.resnet101(pretrained=True)
            self.feature_channel = 2048
        elif kwargs['NUM_LAYERS'] == 50:
            x = tm.resnet50(pretrained=True)
            self.feature_channel = 2048
        elif kwargs['NUM_LAYERS'] == 34:
            x = tm.resnet34(pretrained=True)
            self.feature_channel = 512
        elif kwargs['NUM_LAYERS'] == 18:
            x = tm.resnet18(pretrained=True)
            self.feature_channel = 512
        else:
            raise NotImplementedError
        model_state = self.preact.state_dict()
        state = {k: v for k, v in x.state_dict().items()
                 if k in self.preact.state_dict() and v.size() == self.preact.state_dict()[k].size()}
        model_state.update(state)
        self.preact.load_state_dict(model_state)

        self.deconv_layers = self._make_deconv_layer()
        #self.final_layer = nn.Conv2d(self.deconv_dim[2], self.num_joints * self.depth_dim, kernel_size=1, stride=1, padding=0)

        h36m_jregressor = np.load('./model_files/J_regressor_h36m.npy')
        self.smpl = SMPL_layer(
            './model_files/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl',
            h36m_jregressor=h36m_jregressor,
            dtype=self.smpl_dtype,
            num_joints=self.num_joints
        )

        self.joint_pairs_24 = ((1, 2), (4, 5), (7, 8),
                               (10, 11), (13, 14), (16, 17), (18, 19), (20, 21), (22, 23))
        self.leaf_pairs = ((0, 1), (3, 4))
        self.root_idx_24 = 0

        # mean shape
        init_shape = np.load('./model_files/h36m_mean_beta.npy')
        self.register_buffer(
            'init_shape',
            torch.Tensor(init_shape).float())

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(self.feature_channel, 1024)
        self.drop1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(1024, 1024)
        self.drop2 = nn.Dropout(p=0.5)
        self.decshape = nn.Linear(1024, 10)
        self.decphi = nn.Linear(1024, 23 * 2)  # [cos(phi), sin(phi)]
        self.decleaf = nn.Linear(1024, 5 * 4)  # rot_mat quat

        # depth model
        depth_params = DepthParams(max_depth=kwargs['DEPTH']['MAX_DEPTH'],dataset=kwargs['DEPTH']['DATASET'],encoder=kwargs['DEPTH']['ENCODER'])
        self.depth_in_channels = kwargs['FUSION']['DEPTH_IN_CHANNELS']

        self.depth_mask = []
        for _ in range(5-self.depth_in_channels):
            self.depth_mask.append(0)
        for _ in range(self.depth_in_channels):
            self.depth_mask.append(1)
        self.depth_mask = torch.tensor(self.depth_mask, requires_grad=False).reshape(1,5,1,1)

        self.depth_encoder = bts_encoder(depth_params)
        self.depth_decoder = bts_decoder(depth_params, self.depth_encoder.feat_out_channels, kwargs['DEPTH']['BTS_SIZE'])

        if kwargs['DEPTH']['ADAPTATION'] == 'conv1x1':
            print('build depth adaptation layers')
            self.depth_input_adapter = nn.Sequential(nn.Conv2d(3,3,kernel_size=1),
                                                     nn.BatchNorm2d(3),
                                                     nn.ReLU(inplace=True))
            self.depth_output_adapter = nn.Sequential(nn.Conv2d(self.depth_in_channels,self.depth_in_channels,kernel_size=1))
        else:
            self.depth_input_adapter = None
            self.depth_output_adapter = None

        self.fusion_module = PoseDepthFusion2D(pose_in_channels=256,
                           depth_in_channels=self.depth_in_channels,
                           out_channels=self.num_joints * self.depth_dim,
                           norm=kwargs['FUSION']['DEPTH_NORM'],
                           fusion_method=kwargs['FUSION']['FUSION_METHOD'],
                           concat_original_feature=kwargs['FUSION']['CONCAT_ORIGINAL_FEATURE'],
                           concat_depth_feature=kwargs['FUSION']['CONCAT_DEPTH_FEATURE'])


    def _make_deconv_layer(self):
        deconv_layers = []
        deconv1 = nn.ConvTranspose2d(
            self.feature_channel, self.deconv_dim[0], kernel_size=4, stride=2, padding=int(4 / 2) - 1, bias=False)
        bn1 = self._norm_layer(self.deconv_dim[0])
        deconv2 = nn.ConvTranspose2d(
            self.deconv_dim[0], self.deconv_dim[1], kernel_size=4, stride=2, padding=int(4 / 2) - 1, bias=False)
        bn2 = self._norm_layer(self.deconv_dim[1])
        deconv3 = nn.ConvTranspose2d(
            self.deconv_dim[1], self.deconv_dim[2], kernel_size=4, stride=2, padding=int(4 / 2) - 1, bias=False)
        bn3 = self._norm_layer(self.deconv_dim[2])

        deconv_layers.append(deconv1)
        deconv_layers.append(bn1)
        deconv_layers.append(nn.ReLU(inplace=True))
        deconv_layers.append(deconv2)
        deconv_layers.append(bn2)
        deconv_layers.append(nn.ReLU(inplace=True))
        deconv_layers.append(deconv3)
        deconv_layers.append(bn3)
        deconv_layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*deconv_layers)

    def init_weights(self, cfg):
        if cfg.MODEL.PRETRAINED:
            print(f'Loading model from {cfg.MODEL.PRETRAINED}...')
            self.load_state_dict(torch.load(cfg.MODEL.PRETRAINED))
        elif cfg.MODEL.TRY_LOAD:
            print(f'Loading model from {cfg.MODEL.TRY_LOAD}...')
            pretrained_state = torch.load(cfg.MODEL.TRY_LOAD)
            model_state = self.state_dict()
            pretrained_state = {k: v for k, v in pretrained_state.items()
                                if k in model_state and v.size() == model_state[k].size()}

            model_state.update(pretrained_state)
            self.load_state_dict(model_state)
        else:
            print('Create new model')
            print('=> init weights')
            self._initialize(conv_init_method='normal_')

            # separate loading if provided
            if cfg.MODEL.HYBRIK_PRETRAINED:
                print(f'Loading hybrik module from {cfg.MODEL.HYBRIK_PRETRAINED}...')
                pretrained_state = torch.load(cfg.MODEL.HYBRIK_PRETRAINED)
                model_state = self.state_dict()
                pretrained_state = {k: v for k, v in pretrained_state.items()
                                    if k in model_state and v.size() == model_state[k].size()}
                print(f'update {len(pretrained_state)} items')
                model_state.update(pretrained_state)
                self.load_state_dict(model_state)

            if cfg.MODEL.DEPTH.PRETRAINED:
                print(f'Loading depth module from {cfg.MODEL.DEPTH.PRETRAINED}...')
                pretrained_state = torch.load(cfg.MODEL.DEPTH.PRETRAINED)
                model_state = self.state_dict()
                pretrained_state = {k: v for k, v in pretrained_state.items()
                                    if k in model_state and v.size() == model_state[k].size()}
                print(f'update {len(pretrained_state)} items')
                model_state.update(pretrained_state)
                self.load_state_dict(model_state)

    def _initialize(self,conv_init_method='normal_'):
        conv_init = eval(f'nn.init.{conv_init_method}')
        for name, m in self.deconv_layers.named_modules():
            if isinstance(m, nn.ConvTranspose2d):
                if conv_init_method == 'normal_':
                    conv_init(m.weight, std=0.001)
                else:
                    conv_init(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        modules = [self.fusion_module.modules()]
        if self.depth_input_adapter is not None:
            modules.append(self.depth_input_adapter.modules())
        if self.depth_output_adapter is not None:
            modules.append(self.depth_output_adapter.modules())

        for module in modules:
            for m in module:
                if isinstance(m, nn.Conv2d):
                    if conv_init_method == 'normal_':
                        conv_init(m.weight, std=0.001)
                    else:
                        conv_init(m.weight)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def freeze_weights(self, freeze_depth_enc=True, freeze_depth_dec=True):
        print('Freezing weights')
        if freeze_depth_enc:
            for param in self.depth_encoder.parameters():
                param.requires_grad = False

        if freeze_depth_dec:
            for param in self.depth_decoder.parameters():
                param.requires_grad = False

        # return trainable params
        trainable_params = []
        for name, p in self.named_parameters():
            if freeze_depth_enc and 'depth_encoder.' in name:
                continue
            if freeze_depth_dec and 'depth_decoder.' in name:
                continue
            trainable_params.append(p)
        return trainable_params

    def uvd_to_cam(self, uvd_jts, trans_inv, intrinsic_param, joint_root, depth_factor, return_relative=True):
        assert uvd_jts.dim() == 3 and uvd_jts.shape[2] == 3, uvd_jts.shape
        uvd_jts_new = uvd_jts.clone()
        assert torch.sum(torch.isnan(uvd_jts)) == 0, ('uvd_jts', uvd_jts)

        # remap uv coordinate to input space
        uvd_jts_new[:, :, 0] = (uvd_jts[:, :, 0] + 0.5) * self.width_dim * 4
        uvd_jts_new[:, :, 1] = (uvd_jts[:, :, 1] + 0.5) * self.height_dim * 4
        # remap d to mm
        uvd_jts_new[:, :, 2] = uvd_jts[:, :, 2] * depth_factor
        assert torch.sum(torch.isnan(uvd_jts_new)) == 0, ('uvd_jts_new', uvd_jts_new)

        dz = uvd_jts_new[:, :, 2]

        # transform in-bbox coordinate to image coordinate
        uv_homo_jts = torch.cat(
            (uvd_jts_new[:, :, :2], torch.ones_like(uvd_jts_new)[:, :, 2:]),
            dim=2)
        # batch-wise matrix multipy : (B,1,2,3) * (B,K,3,1) -> (B,K,2,1)
        uv_jts = torch.matmul(trans_inv.unsqueeze(1), uv_homo_jts.unsqueeze(-1))
        # transform (u,v,1) to (x,y,z)
        cam_2d_homo = torch.cat(
            (uv_jts, torch.ones_like(uv_jts)[:, :, :1, :]),
            dim=2)
        # batch-wise matrix multipy : (B,1,3,3) * (B,K,3,1) -> (B,K,3,1)
        xyz_jts = torch.matmul(intrinsic_param.unsqueeze(1), cam_2d_homo)
        xyz_jts = xyz_jts.squeeze(dim=3)
        # recover absolute z : (B,K) + (B,1)
        abs_z = dz + joint_root[:, 2].unsqueeze(-1)
        # multipy absolute z : (B,K,3) * (B,K,1)
        xyz_jts = xyz_jts * abs_z.unsqueeze(-1)

        if return_relative:
            # (B,K,3) - (B,1,3)
            xyz_jts = xyz_jts - joint_root.unsqueeze(1)

        xyz_jts = xyz_jts / depth_factor.unsqueeze(-1)

        return xyz_jts

    def flip_uvd_coord(self, pred_jts, shift=False, flatten=True):
        num_joints = 24

        if flatten:
            assert pred_jts.dim() == 2
            num_batches = pred_jts.shape[0]
            pred_jts = pred_jts.reshape(num_batches, num_joints, 3)
        else:
            assert pred_jts.dim() == 3
            num_batches = pred_jts.shape[0]

        # flip
        if shift:
            pred_jts[:, :, 0] = - pred_jts[:, :, 0]
        else:
            pred_jts[:, :, 0] = -1 / self.width_dim - pred_jts[:, :, 0]

        for pair in self.joint_pairs_24:
            dim0, dim1 = pair
            idx = torch.Tensor((dim0, dim1)).long()
            inv_idx = torch.Tensor((dim1, dim0)).long()
            pred_jts[:, idx] = pred_jts[:, inv_idx]

        if flatten:
            pred_jts = pred_jts.reshape(num_batches, num_joints * 3)

        return pred_jts

    def flip_phi(self, pred_phi):
        pred_phi[:, :, 1] = -1 * pred_phi[:, :, 1]

        for pair in self.joint_pairs_24:
            dim0, dim1 = pair
            idx = torch.Tensor((dim0 - 1, dim1 - 1)).long()
            inv_idx = torch.Tensor((dim1 - 1, dim0 - 1)).long()
            pred_phi[:, idx] = pred_phi[:, inv_idx]

        return pred_phi

    def flip_leaf(self, pred_leaf):

        pred_leaf[:, :, 2] = -1 * pred_leaf[:, :, 2]
        pred_leaf[:, :, 3] = -1 * pred_leaf[:, :, 3]

        for pair in self.leaf_pairs:
            dim0, dim1 = pair
            idx = torch.Tensor((dim0, dim1)).long()
            inv_idx = torch.Tensor((dim1, dim0)).long()
            pred_leaf[:, idx] = pred_leaf[:, inv_idx]

        return pred_leaf

    # x.shape: [batch_size, 3, 256, 256]
    def forward(self, x, trans_inv, intrinsic_param, joint_root, depth_factor, flip_item=None, flip_output=False):
        batch_size = x.shape[0]

        # run depth module
        focal = 518.8579 # nyu focal
        # d_feat[0].shape: [32, 96, 128, 128]
        # d_feat[1].shape: [32, 96, 64, 64]
        # d_feat[2].shape: [32, 192, 32, 32]
        # d_feat[3].shape: [32, 384, 16, 16]
        # d_feat[4].shape: [32, 2208, 8, 8]
        x_depth = None
        if self.depth_input_adapter is None:
            x_depth = x
        else:
            x_depth = self.depth_input_adapter(x) # input adaptation
        d_feat = self.depth_encoder(x_depth)

        # dec_d: 5 x [32, 1, 256, 256]
        # lpg8x8, lpg4x4, lpg2x2, reduc1x1, depth_est
        depth_outputs = self.depth_decoder(d_feat, focal)
        depth_f = torch.cat(depth_outputs[-self.depth_in_channels:],1)

        #depth_f = torch.mul(depth_f, self.depth_mask.to(depth_f.device)) # choose depth features by self.depth_in_channels

        if self.depth_output_adapter is not None:
            depth_f = self.depth_output_adapter(depth_f) # output adaptation

        x0 = self.preact(x) # [32, 512, 8, 8]
        out = self.deconv_layers(x0) # [32, 256, 64, 64]
        #out = self.final_layer(out) # [32, 1536, 64, 64]

        # fusion module
        out = self.fusion_module(out, depth_f)

        out = out.reshape((out.shape[0], self.num_joints, -1))
        out = norm_heatmap(self.norm_type, out)
        assert out.dim() == 3, out.shape

        if self.norm_type == 'sigmoid':
            maxvals, _ = torch.max(out, dim=2, keepdim=True)
        else:
            maxvals = torch.ones((*out.shape[:2], 1), dtype=torch.float, device=out.device)

        heatmaps = out / out.sum(dim=2, keepdim=True)

        heatmaps = heatmaps.reshape((heatmaps.shape[0], self.num_joints, self.depth_dim, self.height_dim, self.width_dim))

        #print(f'heatmaps.shape: {heatmaps.shape}') # [32, 24, 64, 64, 64]

        hm_x = heatmaps.sum((2, 3))
        hm_y = heatmaps.sum((2, 4))
        hm_z = heatmaps.sum((3, 4))

        #print(f'hm_x.shape: {hm_x.shape}') # [32, 24, 64]
        #print(f'hm_y.shape: {hm_y.shape}') # [32, 24, 64]
        #print(f'hm_z.shape: {hm_z.shape}') # [32, 24, 64]


        """
        hm_x = hm_x * torch.cuda.comm.broadcast(torch.arange(hm_x.shape[-1]).type(
            torch.cuda.FloatTensor), devices=[hm_x.device.index])[0]
        hm_y = hm_y * torch.cuda.comm.broadcast(torch.arange(hm_y.shape[-1]).type(
            torch.cuda.FloatTensor), devices=[hm_y.device.index])[0]
        hm_z = hm_z * torch.cuda.comm.broadcast(torch.arange(hm_z.shape[-1]).type(
            torch.cuda.FloatTensor), devices=[hm_z.device.index])[0]
        """

        hm_x = hm_x * torch.arange(hm_x.shape[-1]).type(
            torch.cuda.FloatTensor).to(hm_x.device.index)
        hm_y = hm_y * torch.arange(hm_y.shape[-1]).type(
            torch.cuda.FloatTensor).to(hm_y.device.index)
        hm_z = hm_z * torch.arange(hm_z.shape[-1]).type(
            torch.cuda.FloatTensor).to(hm_z.device.index)

        coord_x = hm_x.sum(dim=2, keepdim=True)
        coord_y = hm_y.sum(dim=2, keepdim=True)
        coord_z = hm_z.sum(dim=2, keepdim=True)

        coord_x = coord_x / float(self.width_dim) - 0.5
        coord_y = coord_y / float(self.height_dim) - 0.5
        coord_z = coord_z / float(self.depth_dim) - 0.5

        #print(f'coord_x.shape: {coord_x.shape}') # [32, 24, 1]
        #print(f'coord_y.shape: {coord_y.shape}') # [32, 24, 1]
        #print(f'coord_z.shape: {coord_z.shape}') # [32, 24, 1]

        #  -0.5 ~ 0.5
        pred_uvd_jts_24 = torch.cat((coord_x, coord_y, coord_z), dim=2)

        pred_uvd_jts_24_flat = pred_uvd_jts_24.reshape((batch_size, self.num_joints * 3))

        x0 = self.avg_pool(x0)
        x0 = x0.view(x0.size(0), -1)
        init_shape = self.init_shape.expand(batch_size, -1)     # (B, 10,)

        xc = x0

        #print(f'xc.shape: {xc.shape}') # [32, 512]

        xc = self.fc1(xc)
        xc = self.drop1(xc)
        xc = self.fc2(xc)
        xc = self.drop2(xc)

        #print(f'xc.shape after drop2: {xc.shape}') # [32, 1024]

        delta_shape = self.decshape(xc)
        pred_shape = delta_shape + init_shape
        pred_phi = self.decphi(xc)
        pred_leaf = self.decleaf(xc)

        #print(f'pred_shape.shape: {pred_shape.shape}') # [32, 10]
        #print(f'pred_phi.shape: {pred_phi.shape}') # [32, 46]
        #print(f'pred_leaf.shape: {pred_leaf.shape}') # [32, 20]

        if flip_item is not None:
            assert flip_output
            pred_uvd_jts_24_orig, pred_phi_orig, pred_leaf_orig, pred_shape_orig = flip_item

        if flip_output:
            pred_uvd_jts_24 = self.flip_uvd_coord(pred_uvd_jts_24, flatten=False, shift=True)
        if flip_output and flip_item is not None:
            pred_uvd_jts_24 = (pred_uvd_jts_24 + pred_uvd_jts_24_orig.reshape(batch_size, 24, 3)) / 2

        pred_uvd_jts_24_flat = pred_uvd_jts_24.reshape((batch_size, self.num_joints * 3))

        #print(f'pred_uvd_jts_24_flat.shape: {pred_uvd_jts_24_flat.shape}') # [32, 72]

        #  -0.5 ~ 0.5
        # Rotate back
        pred_xyz_jts_24 = self.uvd_to_cam(pred_uvd_jts_24[:, :24, :], trans_inv, intrinsic_param, joint_root, depth_factor)
        assert torch.sum(torch.isnan(pred_xyz_jts_24)) == 0, ('pred_xyz_jts_24', pred_xyz_jts_24)

        pred_xyz_jts_24 = pred_xyz_jts_24 - pred_xyz_jts_24[:, self.root_idx_24, :].unsqueeze(1)

        #print(f'pred_xyz_jts_24.shape: {pred_xyz_jts_24.shape}') # [32, 24, 3]

        pred_phi = pred_phi.reshape(batch_size, 23, 2)
        pred_leaf = pred_leaf.reshape(batch_size, 5, 4)

        if flip_output:
            pred_phi = self.flip_phi(pred_phi)
            pred_leaf = self.flip_leaf(pred_leaf)
        if flip_output and flip_item is not None:
            pred_phi = (pred_phi + pred_phi_orig) / 2
            pred_leaf = (pred_leaf + pred_leaf_orig) / 2
            pred_shape = (pred_shape + pred_shape_orig) / 2

        output = self.smpl.hybrik(
            pose_skeleton=pred_xyz_jts_24.type(self.smpl_dtype) * 2,
            betas=pred_shape.type(self.smpl_dtype),
            phis=pred_phi.type(self.smpl_dtype),
            leaf_thetas=pred_leaf.type(self.smpl_dtype),
            global_orient=None,
            return_verts=True
        )
        pred_vertices = output.vertices.float()
        #  -0.5 ~ 0.5
        pred_xyz_jts_24_struct = output.joints.float() / 2
        #  -0.5 ~ 0.5
        pred_xyz_jts_17 = output.joints_from_verts.float() / 2
        pred_theta_mats = output.rot_mats.float().reshape(batch_size, 24 * 4)
        pred_xyz_jts_24 = pred_xyz_jts_24.reshape(batch_size, 72)
        pred_xyz_jts_24_struct = pred_xyz_jts_24_struct.reshape(batch_size, 72)
        pred_xyz_jts_17 = pred_xyz_jts_17.reshape(batch_size, 17 * 3)

        #print(f'pred_theta_mats.shape: {pred_theta_mats.shape}') # [32, 96]
        #print(f'pred_xyz_jts_24.shape: {pred_xyz_jts_24.shape}') # [32, 72]
        #print(f'pred_xyz_jts_24_struct.shape: {pred_xyz_jts_24_struct.shape}') # [32, 72]
        #print(f'pred_xyz_jts_17.shape: {pred_xyz_jts_17.shape}') # [32, 51]

        output = ModelOutput(
            pred_phi=pred_phi,
            pred_leaf=pred_leaf,
            pred_delta_shape=delta_shape,
            pred_shape=pred_shape,
            pred_theta_mats=pred_theta_mats,
            pred_uvd_jts=pred_uvd_jts_24_flat,
            pred_xyz_jts_24=pred_xyz_jts_24,
            pred_xyz_jts_24_struct=pred_xyz_jts_24_struct,
            pred_xyz_jts_17=pred_xyz_jts_17,
            pred_vertices=pred_vertices,
            maxvals=maxvals
        )
        return output

    def forward_gt_theta(self, gt_theta, gt_beta):

        output = self.smpl(
            pose_axis_angle=gt_theta,
            betas=gt_beta,
            global_orient=None,
            return_verts=True
        )

        return output
