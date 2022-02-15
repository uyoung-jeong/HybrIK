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

    def forward(self, pose_f, depth_f):
        batch_size, pose_in_channels, pose_h, pose_w = pose_f.shape
        _, depth_in_channels, depth_h, depth_w = depth_f.shape

        # fit to pose feature shape
        if (pose_h != depth_h) or (pose_w != depth_w):
            depth_f = F.interpolate(depth_f, (pose_h, pose_w), mode='nearest')

        # normalize
        if self.norm == 'softmax':
            depth_f = F.softmax(depth_f.reshape(batch_size, depth_in_channels, -1), dim=-1)
        depth_f = depth_f.reshape(batch_size, depth_in_channels, pose_h, pose_w)

        # fusion
        fusion_f = None
        if self.fusion_method == 'hadamard':
            fusion_f = []
            for d_i in self.depth_in_channels: # fusion for each depth channel
                fusion_f.append(torch.mul(pose_f, depth_f[:,d_i].unsqueeze(1)))
            fusion_f = torch.cat(fusion_f,1)
        else:
            fusion_f = pose_f

        # concat if needed
        if self.concat_original_feature:
            fusion_f = torch.cat((fusion_f,pose_f),1)

        if self.concat_depth_feature:
            fusion_f = torch.cat((fusion_f,depth_f),1)

        # final layer
        out = self.final_layer(fusion_f)

        return out
