import importlib.util
import os
from pathlib import Path
import sys

import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

SIGREG_KERNEL_REPO = "mlengineer-ai/sigreg-characteristic-statistic"
PROJECTION_NORMALIZATION_KERNEL_REPO = "mlengineer-ai/projection-column-normalization"

def modulate(x, shift, scale):
    """AdaLN-zero modulation"""
    return x * (1 + scale) + shift

class SIGReg(torch.nn.Module):
    """Sketch Isotropic Gaussian Regularizer (single-GPU!)"""

    def __init__(
        self,
        knots=17,
        num_proj=1024,
        implementation="eager",
        kernel_repo_id=SIGREG_KERNEL_REPO,
        validation_steps=0,
        normalization_implementation="eager",
        normalization_kernel_repo_id=PROJECTION_NORMALIZATION_KERNEL_REPO,
        normalization_validation_steps=0,
    ):
        super().__init__()
        if implementation not in {"eager", "fused", "validate"}:
            raise ValueError(f"Unknown SIGReg implementation: {implementation}")
        if normalization_implementation not in {"eager", "fused", "validate"}:
            raise ValueError(
                f"Unknown projection normalization implementation: {normalization_implementation}"
            )
        self.num_proj = num_proj
        self.implementation = implementation
        self.kernel_repo_id = kernel_repo_id
        self.validation_steps = validation_steps
        self.normalization_implementation = normalization_implementation
        self.normalization_kernel_repo_id = normalization_kernel_repo_id
        self.normalization_validation_steps = normalization_validation_steps
        self.validation_calls = 0
        self.normalization_validation_calls = 0
        self.comparison_active = False
        self.normalization_comparison_active = False
        self.validation_eager_loss = None
        self._contract_reported = False
        self._kernel = None
        self._normalization_contract_reported = False
        self._normalization_kernel = None
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def _eager_statistic(self, y):
        angles = y.float().unsqueeze(-1) * self.t
        err = (angles.cos().mean(0) - self.phi).square() + angles.sin().mean(0).square()
        return y.size(0) * (err * self.weights).sum() / (y.size(1) * y.size(2))

    def _load_kernel(self):
        if self._kernel is None:
            overrides = dict(
                entry.split("=", 1)
                for entry in os.environ.get("LOCAL_KERNELS", "").split(os.pathsep)
                if "=" in entry
            )
            kernel_path = Path(overrides.get(self.kernel_repo_id, ""))
            init_path = kernel_path / "__init__.py"
            if not init_path.is_file():
                raise RuntimeError(
                    f"No local SIGReg kernel package for {self.kernel_repo_id!r}; "
                    f"expected {init_path}"
                )
            module_name = "lewm_sigreg_characteristic_statistic"
            if module_name in sys.modules:
                self._kernel = sys.modules[module_name]
                return self._kernel
            spec = importlib.util.spec_from_file_location(
                module_name,
                init_path,
                submodule_search_locations=[str(kernel_path)],
            )
            if spec is None or spec.loader is None:
                raise RuntimeError(f"Cannot load local SIGReg kernel package from {init_path}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            self._kernel = module
        return self._kernel

    def _load_normalization_kernel(self):
        if self._normalization_kernel is None:
            overrides = dict(
                entry.split("=", 1)
                for entry in os.environ.get("LOCAL_KERNELS", "").split(os.pathsep)
                if "=" in entry
            )
            kernel_path = Path(overrides.get(self.normalization_kernel_repo_id, ""))
            init_path = kernel_path / "__init__.py"
            if not init_path.is_file():
                raise RuntimeError(
                    "No local projection normalization kernel package for "
                    f"{self.normalization_kernel_repo_id!r}; expected {init_path}"
                )
            module_name = "lewm_projection_column_normalization"
            if module_name in sys.modules:
                self._normalization_kernel = sys.modules[module_name]
                return self._normalization_kernel
            spec = importlib.util.spec_from_file_location(
                module_name,
                init_path,
                submodule_search_locations=[str(kernel_path)],
            )
            if spec is None or spec.loader is None:
                raise RuntimeError(
                    f"Cannot load local projection normalization kernel from {init_path}"
                )
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            self._normalization_kernel = module
        return self._normalization_kernel

    def _normalize_projection_columns(self, projection):
        if self.normalization_implementation == "eager":
            return projection.div_(projection.norm(p=2, dim=0))
        kernel = self._load_normalization_kernel()
        if not self._normalization_contract_reported:
            print(
                "Projection normalization kernel contract: "
                f"A={tuple(projection.shape)} {projection.dtype} "
                f"contiguous={projection.is_contiguous()}, module={kernel.__file__}"
            )
            self._normalization_contract_reported = True
        return kernel.normalize_projection_columns(projection, eps=0.0)

    def _validate_contract(self, y):
        if y.ndim != 3:
            raise RuntimeError(f"Projected SIGReg input must be (B,T,P), got {tuple(y.shape)}")
        if not y.is_cuda or not y.is_contiguous():
            raise RuntimeError("Projected SIGReg input must be a contiguous CUDA tensor")
        if y.dtype not in {torch.float32, torch.float16, torch.bfloat16}:
            raise RuntimeError(f"Unsupported projected SIGReg dtype: {y.dtype}")
        for name in ("t", "phi", "weights"):
            value = getattr(self, name)
            if value.dtype != torch.float32 or not value.is_contiguous() or value.device != y.device:
                raise RuntimeError(f"SIGReg buffer {name} must be contiguous FP32 on {y.device}")
        if self.t[0].item() != 0.0 or self.phi[0].item() != 1.0:
            raise RuntimeError("Configured SIGReg knot-zero invariant does not hold")
        if not self._contract_reported:
            kernel = self._load_kernel()
            print(
                "SIGReg kernel contract: "
                f"y={tuple(y.shape)} {y.dtype} contiguous={y.is_contiguous()}, "
                f"buffers={tuple(self.t.shape)} {self.t.dtype}, module={kernel.__file__}"
            )
            self._contract_reported = True

    def forward(self, proj, validate=False):
        """
        proj: (T, B, D)
        """
        # sample random projections
        A_raw = torch.randn(proj.size(-1), self.num_proj, device=proj.device, dtype=proj.dtype)
        validate_normalization = (
            self.normalization_implementation == "validate"
            and validate
            and self.normalization_validation_calls < self.normalization_validation_steps
        )
        eager_A = A_raw / A_raw.norm(p=2, dim=0) if validate_normalization else None
        A = self._normalize_projection_columns(A_raw)
        y = (proj @ A).permute(1, 0, 2).contiguous()
        eager_y = (
            (proj @ eager_A).permute(1, 0, 2).contiguous()
            if eager_A is not None
            else None
        )

        self.comparison_active = False
        self.normalization_comparison_active = False
        self.validation_eager_loss = None
        if validate_normalization:
            matrix_diff = (A.float() - eager_A.float()).abs()
            projection_diff = (y.float() - eager_y.float()).abs()
            matrix_max_abs = matrix_diff.max().item()
            projection_max_abs = projection_diff.max().item()
            if matrix_max_abs > 2e-2 or projection_max_abs > 0.25:
                raise RuntimeError(
                    "Projection normalization mismatch: "
                    f"matrix_max_abs={matrix_max_abs}, projection_max_abs={projection_max_abs}"
                )
            self.normalization_validation_calls += 1
            self.normalization_comparison_active = True
            print(
                f"Projection normalization validation {self.normalization_validation_calls}: "
                f"matrix_max_abs={matrix_max_abs}, projection_max_abs={projection_max_abs}"
            )

        if self.implementation == "eager":
            loss = self._eager_statistic(y)
            if validate_normalization:
                self.validation_eager_loss = self._eager_statistic(eager_y)
                self.comparison_active = True
            return loss

        self._validate_contract(y)
        fused_loss = self._load_kernel().sigreg_statistic(y, self.t, self.phi, self.weights)
        should_compare = (
            self.implementation == "validate"
            and validate
            and self.validation_calls < self.validation_steps
        )
        if should_compare:
            self.validation_eager_loss = self._eager_statistic(
                eager_y if eager_y is not None else y
            )
            self.comparison_active = True
            self.validation_calls += 1
        elif validate_normalization:
            self.validation_eager_loss = self._eager_statistic(eager_y)
            self.comparison_active = True
        return fused_loss
    
class FeedForward(nn.Module):
    """FeedForward network used in Transformers"""

    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    """Scaled dot-product attention with causal masking"""

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head**-0.5
        self.dropout = dropout
        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x, causal=True):
        """
        x : (B, T, D)
        """
        x = self.norm(x)
        drop = self.dropout if self.training else 0.0
        qkv = self.to_qkv(x).chunk(3, dim=-1)  # q, k, v: (B, heads, T, dim_head)
        q, k, v = (rearrange(t, "b t (h d) -> b h t d", h=self.heads) for t in qkv)
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=drop, is_causal=causal)
        out = rearrange(out, "b h t d -> b t (h d)")
        return self.to_out(out)


class ConditionalBlock(nn.Module):
    """Transformer block with AdaLN-zero conditioning"""

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()

        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True)
        )

        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class Block(nn.Module):
    """Standard Transformer block"""

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()

        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class Transformer(nn.Module):
    """Standard Transformer with support for AdaLN-zero blocks"""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        depth,
        heads,
        dim_head,
        mlp_dim,
        dropout=0.0,
        block_class=Block,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.layers = nn.ModuleList([])

        self.input_proj = (
            nn.Linear(input_dim, hidden_dim)
            if input_dim != hidden_dim
            else nn.Identity()
        )

        self.cond_proj = (
            nn.Linear(input_dim, hidden_dim)
            if input_dim != hidden_dim
            else nn.Identity()
        )

        self.output_proj = (
            nn.Linear(hidden_dim, output_dim)
            if hidden_dim != output_dim
            else nn.Identity()
        )

        for _ in range(depth):
            self.layers.append(
                block_class(hidden_dim, heads, dim_head, mlp_dim, dropout)
            )

    def forward(self, x, c=None):

        if hasattr(self, "input_proj"):
            x = self.input_proj(x)

        if c is not None and hasattr(self, "cond_proj"):
            c = self.cond_proj(c)

        for block in self.layers:
            x = block(x) if isinstance(block, Block) else block(x, c)
        x = self.norm(x)

        if hasattr(self, "output_proj"):
            x = self.output_proj(x)
        return x

class Embedder(nn.Module):
    def __init__(
        self,
        input_dim=10,
        smoothed_dim=10,
        emb_dim=10,
        mlp_scale=4,
    ):
        super().__init__()
        self.patch_embed = nn.Conv1d(input_dim, smoothed_dim, kernel_size=1, stride=1)
        self.embed = nn.Sequential(
            nn.Linear(smoothed_dim, mlp_scale * emb_dim),
            nn.SiLU(),
            nn.Linear(mlp_scale * emb_dim, emb_dim),
        )

    def forward(self, x):
        """
        x: (B, T, D)
        """
        x = x.float()
        x = x.permute(0, 2, 1)
        x = self.patch_embed(x)
        x = x.permute(0, 2, 1)
        x = self.embed(x)
        return x


class MLP(nn.Module):
    """Simple MLP with optional normalization and activation"""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim=None,
        norm_fn=nn.LayerNorm,
        act_fn=nn.GELU,
    ):
        super().__init__()
        norm_fn = norm_fn(hidden_dim) if norm_fn is not None else nn.Identity()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            norm_fn,
            act_fn(),
            nn.Linear(hidden_dim, output_dim or input_dim),
        )

    def forward(self, x):
        """
        x: (B*T, D)
        """
        return self.net(x)


class ARPredictor(nn.Module):
    """Autoregressive predictor for next-step embedding prediction."""

    def __init__(
        self,
        *,
        num_frames,
        depth,
        heads,
        mlp_dim,
        input_dim,
        hidden_dim,
        output_dim=None,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
    ):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, input_dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(
            input_dim,
            hidden_dim,
            output_dim or input_dim,
            depth,
            heads,
            dim_head,
            mlp_dim,
            dropout,
            block_class=ConditionalBlock,
        )

    def forward(self, x, c):
        """
        x: (B, T, d)
        c: (B, T, act_dim)
        """
        T = x.size(1)
        x = x + self.pos_embedding[:, :T]
        x = self.dropout(x)
        x = self.transformer(x, c)
        return x
