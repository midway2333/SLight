import torch
import torch.nn.functional as fc
from torch import nn, Tensor
from typing import Optional
from torch.utils.tensorboard import SummaryWriter

class Multi_Self_Attn(nn.Module):
    """多头自注意力层"""
    def __init__(self, d, dk, head_num, use_dropout: bool=False):
        """
        参数:
        - d: 输入/输出的维度
        - dk: 每个头的维度
        - head_num: 头的数量
        - use_dropout: 是否使用dropout
        """
        super().__init__()
        self.head_num = head_num
        self.dk = dk
        self.use_dropout = use_dropout

        self.q_proj = nn.Linear(d, head_num * dk)
        self.k_proj = nn.Linear(d, head_num * dk)
        self.v_proj = nn.Linear(d, head_num * dk)
        self.o_proj = nn.Linear(head_num * dk, d)
        # 合并所有头的投影矩阵

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        batch_size, seq_len, _ = x.size()

        Q: Tensor = self.q_proj(x)  
        K: Tensor = self.k_proj(x)
        V: Tensor = self.v_proj(x)
        # 并行投影, 方便并行计算

        Q = Q.view(batch_size, seq_len, self.head_num, self.dk).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.head_num, self.dk).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.head_num, self.dk).transpose(1, 2)
        # 分成多个注意力头 [batch, seq_len, head_num, dk]

        attn_output = fc.scaled_dot_product_attention(Q, K, V,
            dropout_p=0.05 if self.use_dropout else 0,
            attn_mask=mask,
            is_causal=False,
        )   # 注意力计算

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, -1)
        return self.o_proj(attn_output)
        # 合并输出


class FeedForward(nn.Module):
    """全连接层"""
    def __init__(self, d, dff, use_dropout: bool=False):
        """
        参数:
        - d: 输入/输出的维度
        - dff: 前馈网络内部层的维度
        - use_dropout: 是否使用dropout
        """
        super().__init__()
        self.ffn = nn.Sequential(                         # 前馈网络
            nn.Linear(d, dff),                            # 维度变换
            nn.Dropout(0.05 if use_dropout else 0),       # Dropout
            nn.GELU(),                                    # 激活函数
            nn.Linear(dff, d),                            # 维度变换
            nn.Dropout(0.05 if use_dropout else 0)        # Dropout
        )

    def forward(self, inputs: Tensor):
        return self.ffn(inputs)


class Encoder(nn.Module):
    """Transformer Encoder"""
    def __init__(self, d, dk, head_num, use_dropout: bool=False):
        """
        参数:
        - d: 输入/输出的维度
        - dk: 每个头的维度
        - head_num: 头的数量
        - use_dropout: 是否使用dropout
        """
        super().__init__()

        self.attn = Multi_Self_Attn(d, dk, head_num, use_dropout)

        self.ffn = FeedForward(d, 4*d, use_dropout)
        # 前馈网络

        self.self_attn_norm = nn.LayerNorm(d)
        self.ffn_norm = nn.LayerNorm(d)
        # 层归一化

    def forward(self, inputs: Tensor) -> Tensor:
        """
        layer_norm 使用 Pre-LN 结构

        参数:
        - inputs: 输入序列 [batch, target_seq, model_dim]
        """
        # ===== 自注意力阶段 =====
        residual = inputs
        norm_input = self.self_attn_norm(inputs)

        self_attn_output = self.attn(
            norm_input,
        )   # 自注意力计算

        self_attn_output = residual + self_attn_output
        # 残差连接

        # ===== 前馈阶段 =====
        norm_output = self.ffn_norm(self_attn_output)
        final_output = self_attn_output + self.ffn(norm_output)
        # 残差连接

        return final_output
    

class ViT_Transformer(nn.Module):
    """ViT模型"""
    def __init__(self, d, dk, head_num, encoder_num, use_dropout: bool=False,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        """
        参数:
        - d: 输入/输出的维度
        - dk: 每个头的维度
        - head_num: 头的数量
        - encoder_num: 编码器数量
        - use_dropout: 是否使用dropout
        - device: 设备类型
        """
        super().__init__()

        self.d = d
        self.device = device
        self.encoder = Encoder(d, dk, head_num, use_dropout).to(device)

        self.encoders = nn.Sequential()   # 模块容器
        for _ in range(encoder_num):
            self.encoders.append(self.encoder.to(device))
            # 添加encoder_num个编码器

    def rope_encode(self, len:int):   # RoPE位置编码
        """本项目的RoPE是外挂而非嵌入, 嵌入太麻烦了, 我懒()"""
        seq_len = len   # 获得输入序列长度

        pos = torch.arange(seq_len, dtype=torch.float).to(self.device)
        # 创建位置索引

        inv_freq = (1.0 / (10000 ** (torch.arange(0, self.d, 2).float() / self.d))).to(self.device)
        # 计算逆频率
        # 在位置编码中引入多尺度信息,确保编码数值稳定性

        sinusoid_inp = torch.einsum("i,d->id", pos, inv_freq)
        # 使用爱因斯坦求和约定
        # 生成位置编码

        pos_emb = torch.cat([torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)], dim=-1)
        # 连接生成结果,三角位置计算

        return pos_emb

    def forward(self, inputs: Tensor) -> Tensor:
        """
        参数:
        - inputs: 输入序列 [batch, target_seq, model_dim]
        """

        inputs = inputs.to(self.device)

        sequence_len = inputs.shape[1]
        # 获取输入张量的序列长度

        inputs = inputs * self.d**0.5 + self.rope_encode(sequence_len)
        # 将嵌入向量乘以嵌入维度的平方根(防止嵌入值过大)

        attn_output = self.encoders(inputs)
        return attn_output
        # 注意力输出


class PatchEmbedding(nn.Module):
    """分割patches"""
    def __init__(self, img_size=224, patch_size=28, in_chans=3, embed_dim=64):
        """
        参数:
        - img_size: 图像大小 (size * size)
        - patch_size: 分割大小 (正方形)
        - in_chans: 输入通道数
        - embed_dim: 嵌入维度
        """
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.conv = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        # 卷积层

        self.pool = nn.MaxPool2d(kernel_size=2)
        # 池化层
    def forward(self, x: Tensor) -> Tensor:
        """
        参数:
        - x: 输入图像 [batch, in_chans, img_size, img_size]
        """        

        x = self.conv(x)   # [embed_dim, num_patches, num_patches]
        # 通过卷积层分割图像

        x = self.pool(x)   # [embed_dim, num_patches/2, num_patches/2]
        # 通过池化层减小尺寸

        x = x.flatten(2)   # [embed_dim, num_patches*2 / 4]
        x = x.transpose(1, 2)   # [num_patches*2 / 4, embed_dim]
        # 调整尺寸以适应Transformer输入格式

        return x


class MLP(nn.Module):
    """分类输出"""
    def __init__(self, d, class_num):
        super().__init__()

        self.norm = nn.LayerNorm(d)
        self.cate = nn.Linear(d, class_num)

    def forward(self, inputs) -> Tensor:
        x = self.norm(inputs[:, 0])
        output = self.cate(x)
        return output
        

class CNN_final(nn.Module):
    """使用CNN代替MLP作为输出层"""
    def __init__(self):
        super().__init__()
        pass

    def forward(self) -> Tensor:
        pass


class Mix_ViT(nn.Module):
    def __init__(self, d, dk, head_num, encoder_num, img_size, patch_size, in_chans, class_num,
        use_dropout: bool=False, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        """
        总模型

        参数:
        - d: 输入/输出的维度
        - dk: 每个头的维度
        - head_num: 头的数量
        - encoder_num: 编码器数量
        - img_size: 图像大小 (size * size)
        - patch_size: 分割大小 (正方形)
        - in_chans: 输入通道数
        - class_num: 种类总数
        - use_dropout: 是否使用dropout
        - device: 设备类型
        """

        super().__init__()

        self.patch_embed = PatchEmbedding(img_size, patch_size, in_chans, d).to(device)
        self.vit = ViT_Transformer(d, dk, head_num, encoder_num, use_dropout, device).to(device)
        self.mlp = MLP(d, class_num).to(device)
        # 模型初始化

        self.cls_token = nn.Parameter(torch.randn(1, 1, d))
        # 可学习的cls token

        self.device = device

    def forward(self, inputs: Tensor) -> Tensor:
        """
        参数:
        - inputs: 输入图像 [batch, in_chans, img_size, img_size]
        """

        inputs = inputs.to(self.device)
        inputs = self.patch_embed(inputs)   # 分块嵌入

        batch_size = inputs.shape[0]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1).to(self.device)   # [batch, 1, d]
        inputs = torch.cat((cls_tokens, inputs), dim=1).to(self.device)          # [batch, num_patches+1, d]
        # 添加cls token

        inputs = self.vit(inputs)
        output = self.mlp(inputs)
        # transformer

        return output
        # 返回分类结果

if __name__ == '__main__':

    BATCH_SIZE = 4
    IMG_SIZE = 224
    PATCH_SIZE = 32
    IN_CHANS = 3
    # 定义输入参数

    vit_model = Mix_ViT(
        d=64,
        dk=32,
        head_num=4,
        encoder_num=2,
        img_size=IMG_SIZE,
        patch_size=PATCH_SIZE,
        in_chans=IN_CHANS,
        class_num=10,
    )   # 创建模型实例

    # 生成随机输入数据 [batch, channels, height, width]
    dummy_input = torch.randn(BATCH_SIZE, IN_CHANS, IMG_SIZE, IMG_SIZE)

    # 前向传播测试
    output: Tensor = vit_model(dummy_input)
    print("Output shape:", output.shape)  # 应为 [4, 10]

    # 使用TensorBoard可视化架构
    writer = SummaryWriter('logs/vit_architecture')  # 日志目录
    writer.add_graph(vit_model, dummy_input)
    writer.close()
