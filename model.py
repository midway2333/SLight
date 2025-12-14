import torch
import torch.nn.functional as fc
from torch import nn, Tensor
from typing import Optional
from torch.utils.tensorboard.writer import SummaryWriter

class RoPE_Emb(nn.Module):
    """RoPE位置编码"""
    def __init__(self, d: int, max_len: int=4096, device: str | None=None):
        """
        RoPE位置编码, Tower2 技术下放(doge)
        - d: 模型维度
        - max_len: 最大序列长度
        """
        super().__init__()

        self.d = d
        self.max_len = max_len
        self.device = device

        inv_freq = 1.0 / (10000 ** (torch.arange(0, d, 2).float().to(device) / d))
        # 计算频率

        self.register_buffer('inv_freq', inv_freq, persistent=False)
        # 注册频率

        self._get_embedding(inv_freq)
        # 预计算

    def _get_embedding(self, inv_freq):
        """预计算位置编码"""
        len_ids = torch.arange(self.max_len, device=self.device)
        # 序列索引

        freqs = torch.outer(len_ids, inv_freq)
        # 计算频率

        emb = torch.cat((freqs, freqs), dim=-1)
        # 复制频率参数, 使复数对共享相同的频率

        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)
        # 频率缓存

    def forward(self) -> tuple:
        """
        生成RoPE位置编码
        """

        self.cos_cached: Tensor
        self.sin_cached: Tensor

        return (
            self.cos_cached,
            self.sin_cached,
        )

def RoPE_rotate(x: Tensor) -> Tensor:
    """
    RoPE旋转操作
    - x: 输入张量
    """
    x1 = x[..., : x.shape[-1] // 2]   # 取前一半维度
    x2 = x[..., x.shape[-1] // 2 :]   # 取后一半维度
    return torch.cat((-x2, x1), dim=-1)   # 拼接

def RoPE_reshape(x: Tensor) -> Tensor:
    """重塑张量形状"""
    batch, head_num, seq_len, dim = x.shape
    x = x.view(batch, head_num, seq_len, dim//2, 2).transpose(4, 3).reshape(batch, head_num, seq_len, dim)

    return x

def RoPE_apply(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor, pos_ids: Tensor):
    """
    应用RoPE编码
    - q: query
    - k: key
    - cos: RoPE cos
    - sin: RoPE sin
    - pos_ids: 位置索引
    """
    cos = cos[pos_ids].unsqueeze(0).unsqueeze(0)   # 按位置索引选择cos值
    sin = sin[pos_ids].unsqueeze(0).unsqueeze(0)   # 按位置索引选择sin值

    q = RoPE_reshape(q)
    # 重塑 Query

    k = RoPE_reshape(k)
    # 重塑 Key

    q_embed = (q * cos) + (RoPE_rotate(q) * sin)
    k_embed = (k * cos) + (RoPE_rotate(k) * sin)
    # 应用旋转位置编码

    return q_embed, k_embed


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

        self.q_proj = nn.Linear(d, head_num * dk, bias=False)
        self.k_proj = nn.Linear(d, head_num * dk, bias=False)
        self.v_proj = nn.Linear(d, head_num * dk, bias=False)
        self.o_proj = nn.Linear(head_num * dk, d, bias=False)
        # 初始化投影层

        self.rope = RoPE_Emb(dk, max_len=4096, device='cuda' if torch.cuda.is_available() else 'cpu')
        # 初始化 RoPE

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        batch_size, seq_len, _ = x.size()

        pos_ids = torch.arange(seq_len, device=x.device)
        # 生成位置索引 [seq_len]

        Q: Tensor = self.q_proj(x)
        K: Tensor = self.k_proj(x)
        V: Tensor = self.v_proj(x)
        # 并行投影, 方便并行计算

        Q = Q.view(batch_size, seq_len, self.head_num, self.dk).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.head_num, self.dk).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.head_num, self.dk).transpose(1, 2)
        # 分成多个注意力头 [batch, seq_len, head_num, dk]

        cos, sin = self.rope()
        Q, K = RoPE_apply(Q, K, cos, sin, pos_ids)

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
            self.encoders = nn.ModuleList([
            Encoder(d, dk, head_num, use_dropout).to(device)
            for _ in range(encoder_num)
        ])  # 添加encoder_num个编码器

    def forward(self, inputs: Tensor) -> Tensor:
        """
        参数:
        - inputs: 输入序列 [batch, target_seq, model_dim]
        """

        x = inputs.to(self.device)
        for encoder in self.encoders:
            x = encoder(x)
        return x
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

        # x = self.pool(x)   # [embed_dim, num_patches/2, num_patches/2]
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
    """使用 CNN 代替 MLP 作为输出层"""
    def __init__(self):
        super().__init__()
        pass

    def forward(self) -> Tensor:   # type: ignore
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
