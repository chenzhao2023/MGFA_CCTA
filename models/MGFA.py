import torch
from torch import nn

class Channel_3D(nn.Module):
    def __init__(self, c1) -> None:
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = nn.Conv3d(c1, c_, 1)
        self.cv2 = nn.Conv3d(c1, 1, 1)
        self.cv3 = nn.Conv3d(c_, c1, 1)
        self.reshape1 = nn.Flatten(start_dim=-3, end_dim=-1)
        self.reshape2 = nn.Flatten()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(1)
        self.layernorm = nn.LayerNorm([c1, 1, 1, 1])

    def forward(self, x):  # shape(batch, channel, depth, height, width)
        x1 = self.reshape1(self.cv1(x))  # shape(batch, channel/2, depth*height*width)
        x2 = self.softmax(self.reshape2(self.cv2(x)))  # shape(batch, depth*height*width)
        y = torch.matmul(x1, x2.unsqueeze(-1)).unsqueeze(-1).unsqueeze(-1)
        return self.sigmoid(self.layernorm(self.cv3(y))) * x

class Spatial_3D(nn.Module):
    def __init__(self, c1) -> None:
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = nn.Conv3d(c1, c_, 1)
        self.cv2 = nn.Conv3d(c1, c_, 1)
        self.reshape1 = nn.Flatten(start_dim=-3, end_dim=-1)
        self.globalPooling = nn.AdaptiveAvgPool3d(1)
        self.softmax = nn.Softmax(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):  # shape(batch, channel, depth, height, width)
        x1 = self.reshape1(self.cv1(x))  # shape(batch, channel/2, depth*height*width)
        x2 = self.softmax(self.globalPooling(self.cv2(x)).squeeze(-1).squeeze(-1).squeeze(-1))  # shape(batch, channel/2)
        y = torch.bmm(x2.unsqueeze(-2), x1)  # shape(batch, 1, depth*height*width)
        return self.sigmoid(y.view(x.shape[0], 1, x.shape[2], x.shape[3], x.shape[4])) * x

class RFEE_3D(nn.Module):
    def __init__(self, in_channel, parallel=True) -> None:
        super().__init__()
        self.parallel = parallel
        self.channel = Channel_3D(in_channel)
        self.spatial = Spatial_3D(in_channel)

    def forward(self, x):
        if self.parallel:
            return self.channel(x) + self.spatial(x)
        return self.spatial(self.channel(x))




def deconv(in_channels, out_channels):  # This is upsample
    return nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)



def initialize_weights(*models):
    for model in models:
        for m in model.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()








class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, drate=0.2):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)

        self.dropout = nn.Dropout(p=drate)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm3d(out_channels)
            )

    def forward(self, x):
        residual = x
        out = self.conv1(x)

        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)

        out = self.bn2(out)

        out += self.shortcut(residual)
        out = self.relu(out)
        out = self.dropout(out)

        return out




class Decoder3d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Decoder3d, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=False),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=False)
        )

    def forward(self, x):
        out = self.conv(x)
        return out
class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Decoder, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1, padding=0),

        )

    def forward(self, x):
        out = self.conv(x)
        return out


class MSFF(nn.Module):
    def __init__(self, in_channels):
        super(MSFF, self).__init__()
        self.conv4_1 = nn.Conv3d(in_channels * 8, 8, kernel_size=1)
        self.conv3_1 = nn.Conv3d(in_channels * 4, 8, kernel_size=1)
        self.conv2_1 = nn.Conv3d(in_channels * 2, 8, kernel_size=1)
        self.conv1_1 = nn.Conv3d(in_channels, 8, kernel_size=1)
        self.convs1 = nn.Conv3d(16, 8, kernel_size=1)
        self.convs2 = nn.Conv3d(16, 8, kernel_size=1)
        self.convs3 = nn.Conv3d(16, 8, kernel_size=1)
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.conv3 = nn.Conv3d(8, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(8, 8, kernel_size=3, padding=1)
        self.conv1 = nn.Conv3d(8, 8, kernel_size=3, padding=1)
        self.conv_fin = nn.Conv3d(8, 1, kernel_size=1)
        self.sigmod = nn.Sigmoid()

    def forward(self, d4, d3, d2, d1):
        d4_1 = self.conv4_1(d4)
        d3_1 = self.conv3_1(d3)
        d2_1 = self.conv2_1(d2)
        d1_1 = self.conv1_1(d1)
        y3 = self.conv3(self.sigmod(self.convs3(torch.cat([self.up(d4_1), d3_1], dim=1))) * d3_1)
        y2 = self.conv2(self.sigmod(self.convs2(torch.cat([self.up(y3), d2_1], dim=1))) * d2_1)
        y1 = self.conv1(self.sigmod(self.convs1(torch.cat([self.up(y2), d1_1], dim=1))) * d1_1)
        out = self.conv_fin(y1)

        return out








# class MSFA_1(nn.Module):
#     def __init__(self, in_channels):
#         super(MSFA_1, self).__init__()
#         self.conv4_1 = nn.Conv3d(in_channels * 8, 16, kernel_size=1)
#         self.conv3_1 = nn.Conv3d(in_channels * 4, 16, kernel_size=1)
#         self.conv2_1 = nn.Conv3d(in_channels * 2, 16, kernel_size=1)
#         self.conv1_1 = nn.Conv3d(in_channels, 16, kernel_size=1)
#         self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
#         self.conv_fin = nn.Conv3d(16, 1, kernel_size=1)
#
#
#         self.dff3 = DFF(16)
#         self.dff2 = DFF(16)
#         self.dff1 = DFF(16)
#
#     def forward(self, d4, d3, d2, d1):
#         d4_1 = self.conv4_1(d4)
#         d3_1 = self.conv3_1(d3)
#         d2_1 = self.conv2_1(d2)
#         d1_1 = self.conv1_1(d1)
#         y3 = self.dff3(self.up(d4_1),d3_1)
#         y2 = self.dff2(self.up(y3),d2_1)
#         y1 = self.dff1(self.up(y2), d1_1)
#         # y3 = self.conv3(self.sigmod(self.convs3(torch.cat([self.up(d4_1), d3_1], dim=1))) * d3_1)
#         # y2 = self.conv2(self.sigmod(self.convs2(torch.cat([self.up(y3), d2_1], dim=1))) * d2_1)
#         # y1 = self.conv1(self.sigmod(self.convs1(torch.cat([self.up(y2), d1_1], dim=1))) * d1_1)
#         out = self.conv_fin(y1)
#         # out = self.sigmod(out)
#
#         return out






class DualAttention(nn.Module):
    def __init__(self, in_channels):
        super(DualAttention, self).__init__()

        self.q_conv = nn.Conv3d(in_channels, in_channels, kernel_size=(1, 3, 1), padding=(0, 1, 0))
        self.k_conv = nn.Conv3d(in_channels, in_channels, kernel_size=(3, 1, 1), padding=(1, 0, 0))
        self.v_conv = nn.Conv3d(in_channels, in_channels, kernel_size=(1, 1, 3), padding=(0, 0, 1))


        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()



    def forward(self, x):
        q0 = self.q_conv(x).view(x.shape[0], -1, x.shape[2] * x.shape[3] * x.shape[4])
        k0 = self.k_conv(x).view(x.shape[0], -1, x.shape[2] * x.shape[3] * x.shape[4])

        v = self.v_conv(x).view(x.shape[0], -1, x.shape[2] * x.shape[3] * x.shape[4])
        q = self.softmax(torch.matmul(k0.transpose(1, 2),q0))

        out = torch.matmul(v,q)

        out = out.view(x.shape[0],x.shape[1],x.shape[2],x.shape[3],x.shape[4])


        return out


def alternate_stack(tensor1, tensor2, dim=3):
    if tensor1.shape != tensor2.shape:
        raise ValueError("tensor1 and tensor2 must have the same shape")


    tensor1_expanded = tensor1.unsqueeze(-1)
    tensor2_expanded = tensor2.unsqueeze(-1)


    combined = torch.cat((tensor1_expanded, tensor2_expanded), dim=-1)


    permute_order = list(range(combined.ndim))
    permute_order[-2], permute_order[-1] = permute_order[-1], permute_order[-2]
    combined = combined.permute(*permute_order)


    new_shape = list(tensor1.shape)
    new_shape[dim] *= 2
    return combined.reshape(*new_shape)


class WeightedSum(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, tensor1, tensor2):

        if tensor1.shape != tensor2.shape:
            raise ValueError("tensor1 and tensor2 must have the same shape")

        return tensor1 + self.scale * tensor2

class MGFANet(nn.Module):
    def __init__(self, in_channels=16):
        super(MGFANet, self).__init__()
        self.down = nn.MaxPool3d((3, 3, 3), stride=(2, 2, 2), padding=1)

        self.residual1 = ResidualBlock(1, in_channels)

        self.residual2 = ResidualBlock(in_channels, in_channels * 2)
        self.residual3 = ResidualBlock(in_channels * 2, in_channels * 4)
        self.residual4 = ResidualBlock(in_channels * 4, in_channels * 8)
        self.residual5 = ResidualBlock(in_channels * 8, in_channels * 16)

        self.lrvencoder1 = Decoder(1,16)

        self.lrvatt0 = WeightedSum()

        self.lrvencoder2 = Decoder(16, 32)

        self.lrvatt1 = WeightedSum()

        self.lrvencoder3 = Decoder(32, 64)

        self.lrvatt2 = WeightedSum()

        self.lrvencoder4 = Decoder(64, 128)

        self.lrvatt3 = WeightedSum()




        self.decoder4 = Decoder3d(in_channels * 16, in_channels * 8)
        self.decoder3  = Decoder3d(in_channels * 8, in_channels * 4)
        self.decoder2 = Decoder3d(in_channels * 4, in_channels * 2)
        self.decoder1 = Decoder3d(in_channels * 2, in_channels * 1)

        self.out = nn.Conv3d(in_channels, 1, 1)

        self.deconv4 = deconv(256, 128)
        self.deconv3 = deconv(128, 64)
        self.deconv2 = deconv(64, 32)
        self.deconv1 = deconv(32, 16)



        self.msfa = MSFF(16)
        self.att5 = RFEE_3D(256)
        self.att4 = RFEE_3D(128)
        self.att3 = RFEE_3D(64)
        self.att2 = RFEE_3D(32)
        self.att1 = RFEE_3D(16)







        initialize_weights(self)



    def forward(self, x, x1):


        e1 = self.residual1(x)
        lr1 = self.lrvencoder1(x1)


        ed1 = self.down(e1)
        lrd1 = self.down(lr1)
        ed1 = self.lrvatt0(ed1, lrd1)
        ed1=self.att1(ed1)



        e2 = self.residual2(ed1)
        lr2 = self.lrvencoder2(lrd1)

        ed2 = self.down(e2)
        lrd2 = self.down(lr2)
        ed2 = self.lrvatt1(ed2, lrd2)
        ed2 = self.att2(ed2)

        e3 = self.residual3(ed2)
        lr3 = self.lrvencoder3(lrd2)

        ed3 = self.down(e3)
        lrd3 = self.down(lr3)
        ed3 = self.lrvatt2(ed3, lrd3)
        ed3 = self.att3(ed3)

        e4 = self.residual4(ed3)
        lr4 = self.lrvencoder4(lrd3)

        ed4 = self.down(e4)
        lrd4 = self.down(lr4)
        ed4 = self.lrvatt3(ed4, lrd4)
        ed4= self.att4(ed4)

        e5 = self.residual5(ed4)
        e5 = self.att5(e5)



        u4 = torch.cat([e4, self.deconv4(e5)],dim=1)

        u4 = self.decoder4(u4)

        u3 = torch.cat([e3, self.deconv3(u4)], dim=1)

        u3 = self.decoder3(u3)

        u2 = torch.cat([e2, self.deconv2(u3)], dim=1)

        u2 = self.decoder2(u2)

        u1 = torch.cat([e1, self.deconv1(u2)], dim=1)

        u1 = self.decoder1(u1)

        out = self.msfa(u4, u3, u2, u1)

        return out

