import torch
import torch.nn as nn
import torch.nn.functional as F

# output size: [self.num_joints * self.depth_dim x width' x height']. [1536, 64, 64] in baseline implementation
class PoseDepthFusion2D(nn.Module):
    def __init__(self, pose_in_channels=256,
                       depth_in_channels=1,
                       out_channels=24*64,
                       norm='softmax',
                       fusion_method='hadamard',
                       concat_original_feature=False,
                       concat_depth_feature=False):
        super(PoseDepthFusion2D, self).__init__()
        self.pose_in_channels = pose_in_channels
        self.depth_in_channels = depth_in_channels
        self.out_channels = out_channels
        self.norm = norm
        self.fusion_method = fusion_method
        self.concat_original_feature = concat_original_feature
        self.concat_depth_feature = concat_depth_feature

        final_layer_in_channels = pose_in_channels * depth_in_channels
        if concat_original_feature:
            final_layer_in_channels += pose_in_channels
        if concat_depth_feature:
            final_layer_in_channels += depth_in_channels

        self.final_layer = nn.Conv2d(
            final_layer_in_channels, out_channels, kernel_size=1, stride=1, padding=0)

        print(f'final layer of 2D fusion module: {self.final_layer.in_channels}->{self.final_layer.out_channels}')

        self.first_log = True

    def forward(self, pose_f, depth_f):
        batch_size, pose_in_channels, pose_h, pose_w = pose_f.shape
        _, depth_in_channels, depth_h, depth_w = depth_f.shape

        # fit to pose feature shape
        if (pose_h != depth_h) or (pose_w != depth_w):
            if self.first_log:
                print(f'reshape depth feature of {depth_h}x{depth_w} to {pose_h}x{pose_w}')
            depth_f = F.interpolate(depth_f, (pose_h, pose_w), mode='nearest')

        # normalize
        if self.norm == 'softmax':
            depth_f = F.softmax(depth_f.reshape(batch_size, depth_in_channels, -1), dim=-1)
        depth_f = depth_f.reshape(batch_size, depth_in_channels, pose_h, pose_w)

        # fusion
        fusion_f = None
        if self.fusion_method == 'hadamard':
            if self.first_log:
                print(f'hadamard product with {self.depth_in_channels} depth channels')
            fusion_f = []
            for d_i in self.depth_in_channels: # fusion for each depth channel
                fusion_f.append(torch.mul(pose_f, depth_f[:,d_i].unsqueeze(1)))
            fusion_f = torch.cat(fusion_f,1)
        else:
            fusion_f = pose_f

        # concat if needed
        if self.concat_original_feature:
            if self.first_log:
                print('concat with original feature')
            fusion_f = torch.cat((fusion_f,pose_f),1)

        if self.concat_depth_feature:
            if self.first_log:
                print('concat with depth feature')
            fusion_f = torch.cat((fusion_f,depth_f),1)

        # final layer
        out = self.final_layer(fusion_f)

        if self.first_log:
            self.first_log = False
        return out

# output size: [self.num_joints x self.depth_dim x width' x height']. [1536, 64, 64] in baseline implementation
class PoseDepthFusion3D(nn.Module):
    def __init__(self, pose_in_channels=256,
                       depth_in_channels=1,
                       depth_dim=64, # unprojection depth dim
                       out_channels=24,
                       norm='softmax',
                       fusion_method='hadamard',
                       concat_original_feature=False,
                       concat_depth_feature=False):
        super(PoseDepthFusion3D, self).__init__()
        self.pose_in_channels = pose_in_channels
        self.depth_in_channels = depth_in_channels
        self.depth_dim = depth_dim
        self.out_channels = out_channels
        self.norm = norm
        self.fusion_method = fusion_method
        self.concat_original_feature = concat_original_feature
        self.concat_depth_feature = concat_depth_feature

        final_layer_in_channels = pose_in_channels * depth_in_channels
        if concat_original_feature:
            final_layer_in_channels += pose_in_channels
        if concat_depth_feature:
            final_layer_in_channels += depth_in_channels

        self.final_layer = nn.Conv2d(
            final_layer_in_channels, out_channels, kernel_size=1, stride=1, padding=0)

        print(f'final layer of 2D fusion module: {self.final_layer.in_channels}->{self.final_layer.out_channels}')

        self.first_log = True

    # f: B x C x H x W
    # d: B x 1 X H x W
    def backproject_heatmap(self, f, depth, cam=None):
        batch_size, in_channels, h, w = f.shape[0], f.shape[1], f.shape[2], f.shape[3]
        device = f.device
        """
        f: BxCxHxW pose heatmap
        depth: Bx1xHxW depth map
        R: Bx3x3 Camera rotation matrix
        T: Bx3x1 Camera translation parameters
        f: Bx2x1 Camera focal length
        c: Bx2x1 Camera center
        k: Bx3x1 # distortion parameter
        p: Bx2x1 # distortion parameter
        """
        R, T, f, c, k, p = None, None, None, None, None, None
        if cam is not None:
            R = camera['R']
            T = camera['T']
            fx = camera['fx']
            fy = camera['fy']
            f = torch.stack([fx, fy])
            cx = camera['cx']
            cy = camera['cy']
            c = torch.stack([cx, cy])
            k = camera['k']
            p = camera['p']
        if cam is None:
            R = torch.eye(3).reshape(1,3,3).repeat(batch_size, 1, 1).to(device)
            T = torch.zeros(batch_size, 3, 1).to(deivce)
            f = torch.ones(batch_size, 2, 1).to(device)
            c = torch.zeros(batch_size, 2, 1).to(device)

        # 2D coordinate matrix
        ii,jj = torch.meshgrid(torch.arange(h,device=device), torch.arange(w, device=device), indexing='ij')
        coords_2d = torch.zeros(batch_size, h, w, 2) # B x H x W
        coords_2d[:,:,0]=ii.repeat(batch_size,1,1)
        coords_2d[:,:,1]=jj.repeat(batch_size,1,1)

        coords_norm = coords_2d/torch.max(torch.tensor([h,w],device=device)) # normalize

        # PlaneSweepStereo way
        xcam = (coords_norm - c) / f

        # camera distortion correction
        if (k is not None) and (p is not None):
            r = torch.sum(xcam ** 2, dim=-1) # B x H x W
            d = 1 - k[:, 0] * r - k[:, 1] * r * r - k[:, 2] * r * r * r  # B x H x W
            u = xcam[:, 0, :] * d - 2 * p[:, 0] * xcam[:, 0, :] * xcam[:, 1, :] - p[:, 1] * (r + 2 * xcam[:, 0, :] * xcam[:, 0, :])  # B x H x W
            v = xcam[:, 1, :] * d - 2 * p[:, 1] * xcam[:, 0, :] * xcam[:, 1, :] - p[:, 0] * (r + 2 * xcam[:, 1, :] * xcam[:, 1, :])  # B x H x W
            xcam = torch.stack([u, v], dim=-1)  # B x H x W x 2

        # get unprojected 3d coordinates
        xcam = torch.cat([xcam, torch.ones(batch_size, h, w, 1).to(device)],dim=-1) # B x H x W x 3
        d = torch.permute(depth, (0,2,3,1)) # B x H x W x 1
        xcam = xcam * d
        xcam = xcam.permute(0,3,1,2).reshape(batch_size, 3,-1) # B x 3 X (HxW)
        x = torch.bmm(torch.inverse(R),xcam) # B x 3 x (HxW)
        x = x.transpose(1,2).reshape(batch_size, h, w, -1) # B x H x W x 3

        import ipdb; ipdb.set_trace()

        print(f'x axis max:{torch.max(x.reshape(-1,3)[:,0])}')
        print(f'x axis min:{torch.min(x.reshape(-1,3)[:,0])}')
        print(f'y axis max:{torch.max(x.reshape(-1,3)[:,1])}')
        print(f'y axis min:{torch.min(x.reshape(-1,3)[:,1])}')
        print(f'z axis max:{torch.max(x.reshape(-1,3)[:,2])}')
        print(f'z axis min:{torch.min(x.reshape(-1,3)[:,2])}')
        # need to check the range of unprojected features.
        # rescale if needed

        # quantize depth into self.depth_dim
        x[:,:,:,2] = x[:,:,:,2] / torch.max(x[:,:,:,2]) * self.depth_dim

        # 3d voxel grid
        voxel = torch.zeros(batch_size, in_channels, self.depth_dim, h, w)

        # 


        y = y.reshape(batch_size, -1, 2).transpose(1, 2)  # [B, 2, PJ]

        xcam = (y - c) / f  # [B, 2, PJ]

        # === remove camera distortion (approx)
        r = torch.sum(xcam ** 2, dim=1)  # [B, PJ]
        d = 1 - k[:, 0] * r - k[:, 1] * r * r - k[:, 2] * r * r * r  # [B, PJ]
        u = xcam[:, 0, :] * d - 2 * p[:, 0] * xcam[:, 0, :] * xcam[:, 1, :] - p[:, 1] * (r + 2 * xcam[:, 0, :] * xcam[:, 0, :])  # [B, PJ]
        v = xcam[:, 1, :] * d - 2 * p[:, 1] * xcam[:, 0, :] * xcam[:, 1, :] - p[:, 0] * (r + 2 * xcam[:, 1, :] * xcam[:, 1, :])  # [B, PJ]
        xcam = torch.stack([u, v], dim=1)  # [B, 2, PJ]

        xcam = torch.cat([xcam, torch.ones(batch_size, 1, xcam.size(-1)).to(xcam.device)], dim=1)  # [B, 3, PJ]
        xcam = xcam.reshape(batch_size, 3, num_persons, num_joints)  # [B, 3, P, J]
        d = depth.reshape(batch_size, 1, -1, 1)  # [B, 1, 1 or P, 1]
        xcam = xcam * d
        xcam = xcam.reshape(batch_size, 3, -1)  # [B, 3, PJ]

        x = torch.bmm(torch.inverse(R), xcam) + T  # [B, 3, PJ]
        x = x.transpose(1, 2)  # [B, PJ, 3]
        x = x.reshape(batch_size, num_persons, num_joints, 3)  # [B, P, J, 3]

        return x

    def forward(self, pose_f, depth_f, cam=None):
        batch_size, pose_in_channels, pose_h, pose_w = pose_f.shape
        _, depth_in_channels, depth_h, depth_w = depth_f.shape

        # fit to pose feature shape
        if (pose_h != depth_h) or (pose_w != depth_w):
            if self.first_log:
                print(f'reshape depth feature of {depth_h}x{depth_w} to {pose_h}x{pose_w}')
            depth_f = F.interpolate(depth_f, (pose_h, pose_w), mode='nearest')

        # normalize
        if self.norm == 'softmax':
            depth_f = F.softmax(depth_f.reshape(batch_size, depth_in_channels, -1), dim=-1)
        depth_f = depth_f.reshape(batch_size, depth_in_channels, pose_h, pose_w)

        # concat if needed
        if self.concat_original_feature:
            if self.first_log:
                print('concat with original feature')
            fusion_f = torch.cat((fusion_f,pose_f),1)

        if self.concat_depth_feature:
            if self.first_log:
                print('concat with depth feature')
            fusion_f = torch.cat((fusion_f,depth_f),1)

        # unproject
        unproj_f = self.backproject_heatmap(fusion_f, depth, cam)

        # final layer
        out = self.final_layer(unproj_f)

        if self.first_log:
            self.first_log = False
        return out
