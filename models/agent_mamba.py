import torch
import torch.nn as nn
from timm.models.layers import DropPath
from models.MambaVision import MambaVisionMixer, Attention
from models.agent_attention import AgentBlock
from timm.models.vision_transformer import Mlp
from models.GatedCNNBlock import GatedCNNOutMixer
from models.kanmodels.kan import KAN

class MixerMLPAttnMLP(nn.Module):
    """
    一个复合块：MambaVisionMixer → MLP → Self-Attention → MLP
    期望输入:  (B, N, C)
    返回输出:  (B, N, C)
    """
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        layer_scale: float | None = None,
        use_ssm: bool = True,
        mixer_kwargs: dict | None = None,   # 可传入 d_state/d_conv/expand 等
        qkv_bias: bool = True,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        mixer_type='gated',
        head_type='mlp',
        agent = True
    ):
        super().__init__()
        self.use_ssm = use_ssm
        mixer_kwargs = mixer_kwargs or dict(d_state=8, d_conv=3, expand=1)

        # ---- 子层 ----
        self.norm1 = norm_layer(dim)
        if mixer_type == "gated":
            # 给一点默认超参，可从外部通过 mixer_kwargs 覆盖
            gkwargs = dict(expansion_ratio=8 / 3, kernel_size=7, conv_ratio=1.0,
                           drop_path=drop_path, layer_scale=1e-6, norm_layer=norm_layer)
            if mixer_kwargs: gkwargs.update(mixer_kwargs)
            self.mixer = GatedCNNOutMixer(d_model=dim, **gkwargs)
        elif mixer_type == "mamba":
            self.mixer = MambaVisionMixer(d_model=dim)
        else:
            raise ValueError(f"Unknown mixer_type={mixer_type}")

        if head_type == "mlp":
            # 给一点默认超参，可从外部通过 mixer_kwargs 覆盖
            self.head1  = Mlp(in_features=dim, hidden_features=int(dim*mlp_ratio), act_layer=act_layer, drop=drop)
            self.head2 = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        elif head_type == "kan":
            self.head1 = KAN([dim, 64, dim])
            self.head2 = KAN([dim, 64, dim])
        else:
            raise ValueError(f"Unknown mixer_type={head_type}")

        self.norm2 = norm_layer(dim)


        self.norm3 = norm_layer(dim)

        self.attn = AgentBlock(dim=768, num_heads=12) if agent else Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                               qk_norm=False, attn_drop=attn_drop, proj_drop=drop, norm_layer=norm_layer)
        self.norm4 = norm_layer(dim)


        # ---- 残差&DropPath&LayerScale ----
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        use_ls = layer_scale is not None and isinstance(layer_scale, (int, float))
        self.gamma1 = nn.Parameter(layer_scale * torch.ones(dim)) if use_ls else None
        self.gamma2 = nn.Parameter(layer_scale * torch.ones(dim)) if use_ls else None
        self.gamma3 = nn.Parameter(layer_scale * torch.ones(dim)) if use_ls else None
        self.gamma4 = nn.Parameter(layer_scale * torch.ones(dim)) if use_ls else None

    def _res(self, x, y, gamma):
        if gamma is not None:
            y = y * gamma
        return x + self.drop_path(y)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        # x: (B, N, C)
        y = self.mixer(self.norm1(x)) if self.use_ssm else self.norm1(x)
        x = self._res(x, y, self.gamma1)

        if not isinstance(self.head1, Mlp):
            x = x.reshape(-1, x.shape[-1])
            # print(x.shape)
            x = self.head1(x)
            x = x.reshape(b, t, d)
            y = self.norm2(x)
        else:
            y = self.head1(self.norm2(x))

        x = self._res(x, y, self.gamma2)

        y = self.attn(self.norm3(x))
        x = self._res(x, y, self.gamma3)

        if not isinstance(self.head2, Mlp):
            x = x.reshape(-1, x.shape[-1])
            x = self.head2(x).reshape(b, t, d)
            y = self.norm4(x)
        else:
            y = self.head2(self.norm4(x))

        x = self._res(x, y, self.gamma4)
        return x

if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 如果没有 CUDA，就关掉 SSM；有 CUDA 再开
    use_ssm = torch.cuda.is_available()
    print(f'use_ssm: {use_ssm}')
    model = MixerMLPAttnMLP(dim=768, num_heads=12,
                          mixer_type='gated',
                          mixer_kwargs=dict(expansion_ratio=8 / 3, kernel_size=7, conv_ratio=0.5)).to(device)

    x = torch.randn(1, 197, 768, device=device)  # 把输入放到同一设备
    model.eval()
    with torch.no_grad():
        y = model(x)
    print(y.shape)  # 预期: torch.Size([1, 197, 768])

