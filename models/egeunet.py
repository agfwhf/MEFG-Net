import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

from timm.models.layers import trunc_normal_
import math


class DepthWiseConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size=3, padding=1, stride=1, dilation=1):
        super().__init__()
        
        self.conv1 = nn.Conv2d(dim_in, dim_in, kernel_size=kernel_size, padding=padding, 
                      stride=stride, dilation=dilation, groups=dim_in)
        # 动态分组逻辑
        if dim_in >= 4:
            num_groups = 4 if dim_in % 4 == 0 else 2
        else:
            num_groups = 1
        
        self.norm_layer = nn.GroupNorm(num_groups, dim_in)
        self.conv2 = nn.Conv2d(dim_in, dim_out, kernel_size=1)

    def forward(self, x):
        return self.conv2(self.norm_layer(self.conv1(x)))


class LayerNorm(nn.Module):
    r""" From ConvNeXt (https://arxiv.org/pdf/2201.03545.pdf)
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
    

class group_aggregation_bridge(nn.Module):
    def __init__(self, dim_xh, dim_xl, k_size=3, d_list=[1,2,5,7]):
        super().__init__()
        self.pre_project = nn.Conv2d(dim_xh, dim_xl, 1)
        group_size = dim_xl // 2
        self.g0 = nn.Sequential(
            LayerNorm(normalized_shape=group_size+1, data_format='channels_first'),
            nn.Conv2d(group_size + 1, group_size + 1, kernel_size=3, stride=1, 
                      padding=(k_size+(k_size-1)*(d_list[0]-1))//2, 
                      dilation=d_list[0], groups=group_size + 1)
        )
        self.g1 = nn.Sequential(
            LayerNorm(normalized_shape=group_size+1, data_format='channels_first'),
            nn.Conv2d(group_size + 1, group_size + 1, kernel_size=3, stride=1, 
                      padding=(k_size+(k_size-1)*(d_list[1]-1))//2, 
                      dilation=d_list[1], groups=group_size + 1)
        )
        self.g2 = nn.Sequential(
            LayerNorm(normalized_shape=group_size+1, data_format='channels_first'),
            nn.Conv2d(group_size + 1, group_size + 1, kernel_size=3, stride=1, 
                      padding=(k_size+(k_size-1)*(d_list[2]-1))//2, 
                      dilation=d_list[2], groups=group_size + 1)
        )
        self.g3 = nn.Sequential(
            LayerNorm(normalized_shape=group_size+1, data_format='channels_first'),
            nn.Conv2d(group_size + 1, group_size + 1, kernel_size=3, stride=1, 
                      padding=(k_size+(k_size-1)*(d_list[3]-1))//2, 
                      dilation=d_list[3], groups=group_size + 1)
        )
        self.tail_conv = nn.Sequential(
            LayerNorm(normalized_shape=dim_xl * 2 + 4, data_format='channels_first'),
            nn.Conv2d(dim_xl * 2 + 4, dim_xl, 1)
        )
    def forward(self, xh, xl, mask):
        xh = self.pre_project(xh)
        xh = F.interpolate(xh, size=[xl.size(2), xl.size(3)], mode ='bilinear', align_corners=True)
        xh = torch.chunk(xh, 4, dim=1)
        xl = torch.chunk(xl, 4, dim=1)
        x0 = self.g0(torch.cat((xh[0], xl[0], mask), dim=1))
        x1 = self.g1(torch.cat((xh[1], xl[1], mask), dim=1))
        x2 = self.g2(torch.cat((xh[2], xl[2], mask), dim=1))
        x3 = self.g3(torch.cat((xh[3], xl[3], mask), dim=1))
        x = torch.cat((x0,x1,x2,x3), dim=1)
        x = self.tail_conv(x)
        return x


class Grouped_multi_axis_Hadamard_Product_Attention(nn.Module):
    def __init__(self, dim_in, dim_out, x=8, y=8):
        super().__init__()
        
        c_dim_in = dim_in//4
        k_size=3
        pad=(k_size-1) // 2
        
        self.params_xy = nn.Parameter(torch.Tensor(1, c_dim_in, x, y), requires_grad=True)
        nn.init.ones_(self.params_xy)
        self.conv_xy = nn.Sequential(nn.Conv2d(c_dim_in, c_dim_in, kernel_size=k_size, padding=pad, groups=c_dim_in), nn.GELU(), nn.Conv2d(c_dim_in, c_dim_in, 1))

        self.params_zx = nn.Parameter(torch.Tensor(1, 1, c_dim_in, x), requires_grad=True)
        nn.init.ones_(self.params_zx)
        self.conv_zx = nn.Sequential(nn.Conv1d(c_dim_in, c_dim_in, kernel_size=k_size, padding=pad, groups=c_dim_in), nn.GELU(), nn.Conv1d(c_dim_in, c_dim_in, 1))

        self.params_zy = nn.Parameter(torch.Tensor(1, 1, c_dim_in, y), requires_grad=True)
        nn.init.ones_(self.params_zy)
        self.conv_zy = nn.Sequential(nn.Conv1d(c_dim_in, c_dim_in, kernel_size=k_size, padding=pad, groups=c_dim_in), nn.GELU(), nn.Conv1d(c_dim_in, c_dim_in, 1))

        self.dw = nn.Sequential(
                nn.Conv2d(c_dim_in, c_dim_in, 1),
                nn.GELU(),
                nn.Conv2d(c_dim_in, c_dim_in, kernel_size=3, padding=1, groups=c_dim_in)
        )
        
        self.norm1 = LayerNorm(dim_in, eps=1e-6, data_format='channels_first')
        self.norm2 = LayerNorm(dim_in, eps=1e-6, data_format='channels_first')
        
        self.ldw = nn.Sequential(
                nn.Conv2d(dim_in, dim_in, kernel_size=3, padding=1, groups=dim_in),
                nn.GELU(),
                nn.Conv2d(dim_in, dim_out, 1),
        )
        
    def forward(self, x):
        x = self.norm1(x)
        x1, x2, x3, x4 = torch.chunk(x, 4, dim=1)
        B, C, H, W = x1.size()
        #----------xy----------#
        params_xy = self.params_xy
        x1 = x1 * self.conv_xy(F.interpolate(params_xy, size=x1.shape[2:4],mode='bilinear', align_corners=True))
        #----------zx----------#
        x2 = x2.permute(0, 3, 1, 2)
        params_zx = self.params_zx
        x2 = x2 * self.conv_zx(F.interpolate(params_zx, size=x2.shape[2:4],mode='bilinear', align_corners=True).squeeze(0)).unsqueeze(0)
        x2 = x2.permute(0, 2, 3, 1)
        #----------zy----------#
        x3 = x3.permute(0, 2, 1, 3)
        params_zy = self.params_zy
        x3 = x3 * self.conv_zy(F.interpolate(params_zy, size=x3.shape[2:4],mode='bilinear', align_corners=True).squeeze(0)).unsqueeze(0)
        x3 = x3.permute(0, 2, 1, 3)
        #----------dw----------#
        x4 = self.dw(x4)
        #----------concat----------#
        x = torch.cat([x1,x2,x3,x4],dim=1)
        #----------ldw----------#
        x = self.norm2(x)
        x = self.ldw(x)
        return x

class ADPModule(nn.Module):
    """改进后的自适应空洞卷积金字塔模块（集成FADC动态权重）"""
    def __init__(self, in_channels, dilations=[1,2,4,6], reduction_ratio=16):
        super().__init__()
        self.in_channels = in_channels
        self.dilation_rates = dilations
        
        # 改进1：增强特征预处理
        self.pre_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=4),
            nn.GroupNorm(4, in_channels),
            nn.GELU()
        )
        
        # 多膨胀率分支
        self.branches = nn.ModuleList([
            self._make_branch(in_channels, d) for d in self.dilation_rates
        ])
        
        # 改进2：动态权重生成网络（空间自适应）
        self.weight_net = nn.Sequential(
            nn.Conv2d(
                in_channels, 
                max(in_channels // reduction_ratio, 4),  # 输入通道数
                kernel_size=3, 
                padding=1
            ),
            nn.GELU(),
            nn.Conv2d(
                max(in_channels // reduction_ratio, 4),   # 输入通道
                len(self.dilation_rates),                 # 输出通道数=分支数
                kernel_size=1                             # 1x1卷积
            ),
            nn.Softmax(dim=1)
        )
        
        # 残差连接
        self.skip = nn.Conv2d(in_channels, in_channels, 1)

    def _make_branch(self, ch, dilation):
        """构建单个膨胀率分支"""
        return nn.Sequential(
            nn.Conv2d(ch, ch, 3, 
                     padding=dilation, 
                     dilation=dilation,
                     groups=ch),  # 深度可分离卷积
            nn.GroupNorm(4, ch),
            nn.GELU(),
            nn.Conv2d(ch, ch, 1)
        )
        
    def forward(self, x):
        # 特征预处理
        x_in = self.pre_conv(x)
        
        # 并行分支计算
        branch_outs = [branch(x_in) for branch in self.branches]
        
        # 生成动态权重 [B, K, H, W]
        weights = self.weight_net(x_in)
        
        # 加权融合
        stacked = torch.stack(branch_outs, dim=1)  # [B, K, C, H, W]
        weighted = (stacked * weights.unsqueeze(2)).sum(dim=1)
        
        return self.skip(x) + weighted

# ==================== 新增边缘路径模块 ====================
class EdgeEncoder(nn.Module):
    """改进的边缘编码器，使用可学习的边缘检测核"""
    def __init__(self, in_ch=3, ch_list=[4,8,12,16,20,24]):
        super().__init__()
        
        # 定义Sobel算子初始化参数
        sobel_x = torch.tensor([[[[1, 0, -1], 
                                [2, 0, -2],
                                [1, 0, -1]]]], dtype=torch.float32)
        sobel_y = torch.tensor([[[[1, 2, 1], 
                                [0, 0, 0],
                                [-1, -2, -1]]]], dtype=torch.float32)
        
        self.layers = nn.ModuleList()
        for i in range(len(ch_list)):
            # 每个阶段使用不同的卷积配置
            conv = DepthWiseConv2d(
                in_ch if i==0 else ch_list[i-1],
                ch_list[i],
                kernel_size=3 if i<3 else 5,  # 深层使用更大的感受野
                stride=1,
                padding=1 if i<3 else 2
            )
            
            # 特殊初始化第一个卷积层
            if i == 0:
                with torch.no_grad():
                    # 通道分组初始化
                    for j in range(ch_list[i]//2):
                        # 交替设置x和y方向边缘检测核
                        conv.conv1.weight[j*2::ch_list[i]] = sobel_x * (j+1)/ch_list[i]
                        conv.conv1.weight[j*2+1::ch_list[i]] = sobel_y * (j+1)/ch_list[i]
                    # 添加随机扰动保持可学习性
                    conv.conv1.weight.data += torch.randn_like(conv.conv1.weight) * 0.1
                    conv.conv1.weight.requires_grad = True
                    
            self.layers.append(nn.Sequential(
                conv,
                nn.GELU(),
                LayerNorm(ch_list[i], data_format="channels_first")
            ))
            
        # 添加边缘锐化层
        self.sharpen = nn.Sequential(
            nn.Conv2d(ch_list[-1], ch_list[-1], 3, 
                     padding=1, groups=ch_list[-1]),  # Depthwise
            nn.Conv2d(ch_list[-1], ch_list[-1], 1)    # Pointwise
        )

    def forward(self, x):
        features = []
        for i, layer in enumerate(self.layers):
            x = F.max_pool2d(layer(x), 2)
            # 仅在最深层应用锐化
            if i == len(self.layers)-1:  # 最后一层通道数=24
                x = x + self.sharpen(x)
            features.append(x)
            # print("Layer {} output shape: {}".format(i, x.shape))

        return features

class DPAG(nn.Module):
    """增强版双路径注意力门控"""
    def __init__(self, main_ch, edge_ch):
        super().__init__()
        
        # 边缘特征增强
        self.edge_enhance = nn.Sequential(
            DepthWiseConv2d(edge_ch, main_ch, kernel_size=5, padding=2),
            nn.GELU(),
            LayerNorm(main_ch, data_format="channels_first"),
            nn.Conv2d(main_ch, main_ch, 3, padding=1)  # 增加特征融合能力
        )
        
        # 混合注意力机制
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(main_ch*2, main_ch//4, 1),
            nn.GELU(),
            nn.Conv2d(main_ch//4, main_ch, 1),
            nn.Sigmoid()
        )
        
        # 改进的空间注意力
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(main_ch*2, main_ch//4, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(main_ch//4, 1, 7, padding=3),
            nn.Sigmoid()
        )
        
        # 残差缩放因子
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, f_main, f_edge):
        # 边缘特征增强
        f_edge = self.edge_enhance(f_edge)
        # 确保边缘特征与主特征尺寸一致
        if f_edge.shape[2:] != f_main.shape[2:]:
            f_edge = F.interpolate(f_edge, size=f_main.shape[2:], mode='bilinear', align_corners=True)
        
        # 通道注意力
        c_attn = self.channel_attn(torch.cat([f_main, f_edge], dim=1))
        
        # 空间注意力
        s_attn = self.spatial_attn(torch.cat([f_main, f_edge], dim=1))
        
        # 混合注意力
        attn = c_attn * s_attn
        
        # 残差连接（可学习缩放）
        return f_main + self.gamma * (attn * f_edge)
    

class EGEUNet(nn.Module):
    
    def __init__(self, 
                 num_classes=3, 
                 input_channels=3, 
                 c_list=[8,16,24,32,48,64], 
                 bridge=True, gt_ds=True,
                 adp_layers=[1,1,1,1,1,0],  # 添加参数定义
                 adp_dilations=[1,2,4,6],  # 添加参数定义
                 adp_reduction=16,
                 reduction_ratio=16,
                 edge_ch_list=[4, 8, 12, 16, 20, 24],  
                 use_dpag=True,
                 use_edge_supervision=True):
        super().__init__()

        self.bridge = bridge
        self.gt_ds = gt_ds
        self.use_dpag = use_dpag

        # 添加边缘监督头
        self.edge_head = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch, 1, 3, padding=1),
                nn.Sigmoid()
            ) for ch in edge_ch_list
        ])
        
        self.use_edge_supervision = use_edge_supervision

        # ========== 新增边缘路径 ==========
        if self.use_dpag:
            # 边缘编码器
            self.edge_encoder = EdgeEncoder(input_channels, edge_ch_list)
            
            # DPAG融合层
            self.dpag_layers = nn.ModuleList([
                DPAG(c_list[i], edge_ch_list[i]) for i in range(len(c_list))
            ])
            
            # 边缘监督头（仅在use_dpag=True时创建）
            self.edge_head = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(ch, 1, 3, padding=1),
                    nn.Sigmoid()
                ) for ch in edge_ch_list
            ])
        else:
            self.edge_head = None  # 明确设置为None
        # # 添加边缘监督头
        # self.edge_head = nn.ModuleList([
        #     nn.Conv2d(ch, 1, 1) for ch in edge_ch_list
        # ])
        
        self.encoder1 = nn.Sequential(
            nn.Conv2d(input_channels, c_list[0], 3, stride=1, padding=1),
        )
        self.encoder2 =nn.Sequential(
            nn.Conv2d(c_list[0], c_list[1], 3, stride=1, padding=1),
        ) 
        self.encoder3 = nn.Sequential(
            nn.Conv2d(c_list[1], c_list[2], 3, stride=1, padding=1),
        )
        self.encoder4 = nn.Sequential(
            Grouped_multi_axis_Hadamard_Product_Attention(c_list[2], c_list[3]),
        )
        self.encoder5 = nn.Sequential(
            Grouped_multi_axis_Hadamard_Product_Attention(c_list[3], c_list[4]),
        )
        self.encoder6 = nn.Sequential(
            Grouped_multi_axis_Hadamard_Product_Attention(c_list[4], c_list[5]),
        )

        if bridge: 
            self.GAB1 = group_aggregation_bridge(c_list[1], c_list[0])
            self.GAB2 = group_aggregation_bridge(c_list[2], c_list[1])
            self.GAB3 = group_aggregation_bridge(c_list[3], c_list[2])
            self.GAB4 = group_aggregation_bridge(c_list[4], c_list[3])
            self.GAB5 = group_aggregation_bridge(c_list[5], c_list[4])
            print('group_aggregation_bridge was used')
        if gt_ds:
            self.gt_conv1 = nn.Sequential(nn.Conv2d(c_list[4], 1, 1))
            self.gt_conv2 = nn.Sequential(nn.Conv2d(c_list[3], 1, 1))
            self.gt_conv3 = nn.Sequential(nn.Conv2d(c_list[2], 1, 1))
            self.gt_conv4 = nn.Sequential(nn.Conv2d(c_list[1], 1, 1))
            self.gt_conv5 = nn.Sequential(nn.Conv2d(c_list[0], 1, 1))
            print('gt deep supervision was used')
        
        self.decoder1 = nn.Sequential(
            Grouped_multi_axis_Hadamard_Product_Attention(c_list[5], c_list[4]),
        ) 
        self.decoder2 = nn.Sequential(
            Grouped_multi_axis_Hadamard_Product_Attention(c_list[4], c_list[3]),
        ) 
        self.decoder3 = nn.Sequential(
            Grouped_multi_axis_Hadamard_Product_Attention(c_list[3], c_list[2]),
        )  
        self.decoder4 = nn.Sequential(
            nn.Conv2d(c_list[2], c_list[1], 3, stride=1, padding=1),
        )  
        self.decoder5 = nn.Sequential(
            nn.Conv2d(c_list[1], c_list[0], 3, stride=1, padding=1),
        )  
        self.ebn1 = nn.GroupNorm(4, c_list[0])
        self.ebn2 = nn.GroupNorm(4, c_list[1])
        self.ebn3 = nn.GroupNorm(4, c_list[2])
        self.ebn4 = nn.GroupNorm(4, c_list[3])
        self.ebn5 = nn.GroupNorm(4, c_list[4])
        self.dbn1 = nn.GroupNorm(4, c_list[4])
        self.dbn2 = nn.GroupNorm(4, c_list[3])
        self.dbn3 = nn.GroupNorm(4, c_list[2])
        self.dbn4 = nn.GroupNorm(4, c_list[1])
        self.dbn5 = nn.GroupNorm(4, c_list[0])

        self.final = nn.Conv2d(c_list[0], num_classes, kernel_size=1)

        self.apply(self._init_weights)

        # 添加ADP模块配置
        self.adp_modules = nn.ModuleList()
        for idx, use_adp in enumerate(adp_layers):
            if use_adp:
                self.adp_modules.append(
                    ADPModule(
                        c_list[idx],
                        dilations=adp_dilations,
                        reduction_ratio=adp_reduction
                    )
                )
            else:
                self.adp_modules.append(nn.Identity())

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, edge_gt=None):  # 添加edge_gt参数

        # ===== 边缘特征提取 =====
        edge_loss = 0
        edge_preds = []
        # print("edge_gt shape:", edge_gt.shape)  # 假设输出为 torch.Size([4, 3072])
    
        if self.use_dpag:
            # print("Input x shape:", x.shape)  # 应为 [batch, channels, H, W]
            edge_features = self.edge_encoder(x)
            # print("edge_features type:", type(edge_features))  # 应为 torch.Tensor
        
            # 添加空值检查
            if self.edge_head is None:
                raise ValueError("edge_head未初始化，请检查use_dpag参数")
        
            # 验证特征层数与监督头数一致
            if len(edge_features) != len(self.edge_head):
                raise RuntimeError(f"边缘特征层数({len(edge_features)})与监督头数({len(self.edge_head)})不匹配")
        
            for f, head in zip(edge_features, self.edge_head):
                pred = head(f)
                edge_preds.append(pred)
            
                # 计算边缘损失
                if self.training and edge_gt is not None:
                    resized_gt = F.interpolate(edge_gt.float(), 
                                          size=pred.shape[2:], 
                                          mode='bilinear',
                                          align_corners=True)
                    edge_loss += F.binary_cross_entropy(pred, resized_gt)
        
        out = F.gelu(F.max_pool2d(self.ebn1(self.encoder1(x)),2,2))
        out = self.adp_modules[0](out)  # 第1层后接入ADP
        # t1 = out # b, c0, H/2, W/2
        # 新增DPAG融合
        if self.use_dpag:
            out = self.dpag_layers[0](out, edge_features[0])  # 主路径+边缘路径融合
        t1 = out  # 后续保持原有GAB流程

        out = F.gelu(F.max_pool2d(self.ebn2(self.encoder2(out)),2,2))
        out = self.adp_modules[1](out)  # 第2层后接入ADP
        t2 = out # b, c1, H/4, W/4 
        if self.use_dpag:
            out = self.dpag_layers[1](out, edge_features[1])
        t2 = out

        out = F.gelu(F.max_pool2d(self.ebn3(self.encoder3(out)),2,2))
        out = self.adp_modules[2](out)  # 第3层后接入ADP
        # t3 = out # b, c2, H/8, W/8
        if self.use_dpag:
            out = self.dpag_layers[2](out, edge_features[2])
        t3 = out
        
        out = F.gelu(F.max_pool2d(self.ebn4(self.encoder4(out)),2,2))
        out = self.adp_modules[3](out)  # 第4层后接入ADP
        # t4 = out # b, c3, H/16, W/16
        if self.use_dpag:
            out = self.dpag_layers[3](out, edge_features[3])
        t4 = out
        
        out = F.gelu(F.max_pool2d(self.ebn5(self.encoder5(out)),2,2))
        out = self.adp_modules[4](out)  # 第5层后接入ADP
        # t5 = out # b, c4, H/32, W/32
        if self.use_dpag:
            out = self.dpag_layers[4](out, edge_features[4])
        t5 = out
        
        out = F.gelu(self.encoder6(out)) # b, c5, H/32, W/32
        t6 = out
        
        out5 = F.gelu(self.dbn1(self.decoder1(out))) # b, c4, H/32, W/32
        if self.gt_ds: 
            gt_pre5 = self.gt_conv1(out5)
            if self.use_dpag:  # 在GAB前融合解码特征
                out5 = self.dpag_layers[4](out5, edge_features[4])
            t5 = self.GAB5(t6, t5, gt_pre5)
            gt_pre5 = F.interpolate(gt_pre5, scale_factor=32, mode ='bilinear', align_corners=True)
        else: t5 = self.GAB5(t6, t5)
        out5 = torch.add(out5, t5) # b, c4, H/32, W/32
        
        out4 = F.gelu(F.interpolate(self.dbn2(self.decoder2(out5)),scale_factor=(2,2),mode ='bilinear',align_corners=True)) # b, c3, H/16, W/16
        if self.gt_ds: 
            gt_pre4 = self.gt_conv2(out4)
            if self.use_dpag:
                out4 = self.dpag_layers[3](out4, edge_features[3])
            t4 = self.GAB4(t5, t4, gt_pre4)
            gt_pre4 = F.interpolate(gt_pre4, scale_factor=16, mode ='bilinear', align_corners=True)
        else:t4 = self.GAB4(t5, t4)
        out4 = torch.add(out4, t4) # b, c3, H/16, W/16
        
        out3 = F.gelu(F.interpolate(self.dbn3(self.decoder3(out4)),scale_factor=(2,2),mode ='bilinear',align_corners=True)) # b, c2, H/8, W/8
        if self.gt_ds: 
            gt_pre3 = self.gt_conv3(out3)
            if self.use_dpag:  # 在GAB前融合解码特征
                out3 = self.dpag_layers[2](out3, edge_features[2])
            t3 = self.GAB3(t4, t3, gt_pre3)
            gt_pre3 = F.interpolate(gt_pre3, scale_factor=8, mode ='bilinear', align_corners=True)
        else: t3 = self.GAB3(t4, t3)
        out3 = torch.add(out3, t3) # b, c2, H/8, W/8
        
        out2 = F.gelu(F.interpolate(self.dbn4(self.decoder4(out3)),scale_factor=(2,2),mode ='bilinear',align_corners=True)) # b, c1, H/4, W/4
        if self.gt_ds: 
            gt_pre2 = self.gt_conv4(out2)
            if self.use_dpag:  # 在GAB前融合解码特征
                out2 = self.dpag_layers[1](out2, edge_features[1])
            t2 = self.GAB2(t3, t2, gt_pre2)
            gt_pre2 = F.interpolate(gt_pre2, scale_factor=4, mode ='bilinear', align_corners=True)
        else: t2 = self.GAB2(t3, t2)
        out2 = torch.add(out2, t2) # b, c1, H/4, W/4 
        
        out1 = F.gelu(F.interpolate(self.dbn5(self.decoder5(out2)),scale_factor=(2,2),mode ='bilinear',align_corners=True)) # b, c0, H/2, W/2
        if self.gt_ds: 
            gt_pre1 = self.gt_conv5(out1)
            if self.use_dpag:  # 在GAB前融合解码特征
                out1 = self.dpag_layers[0](out1, edge_features[0])
            t1 = self.GAB1(t2, t1, gt_pre1)
            gt_pre1 = F.interpolate(gt_pre1, scale_factor=2, mode ='bilinear', align_corners=True)
        else: t1 = self.GAB1(t2, t1)
        out1 = torch.add(out1, t1) # b, c0, H/2, W/2
        
        out0 = F.interpolate(self.final(out1),scale_factor=(2,2),mode ='bilinear',align_corners=True) # b, num_class, H, W
        
        # 返回结果
        outputs = {
            'segmentation': out0,
            'edge_preds': edge_preds if self.use_dpag else None
        }
        
        if self.training:
            outputs['edge_loss'] = edge_loss
            if self.gt_ds:
                outputs['deep_supervision'] = (gt_pre5, gt_pre4, gt_pre3, gt_pre2, gt_pre1)
        return outputs