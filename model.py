"""
Tiny Recursive Model (TRM) Implementation
Based on "Less is More: Recursive Reasoning with Tiny Networks"

Supports sudoku and maze tasks via pluggable TaskHead modules.
The core recursive mechanism is task-agnostic; only the input encoder and output
decoder (TaskHead) change per task.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, List


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / norm * self.weight


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding"""
    def __init__(self, dim: int, max_seq_len: int = 2048):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, seq_len: int) -> tuple:
        t = torch.arange(seq_len, device=self.inv_freq.device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()


def apply_rotary_pos_emb(q, k, cos, sin):
    """Apply rotary embeddings to queries and keys"""
    def rotate_half(x):
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


class SwiGLU(nn.Module):
    """SwiGLU activation function"""
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class MLPMixer(nn.Module):
    """MLP-Mixer for sequence mixing (fixed context size)"""
    def __init__(self, seq_len: int, dim: int):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.mix = nn.Linear(seq_len, seq_len, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = x.transpose(1, 2)
        x = self.mix(x)
        return x.transpose(1, 2)


class SelfAttention(nn.Module):
    """Multi-head self-attention with rotary embeddings (bidirectional)"""
    def __init__(self, dim: int, n_heads: int = 8):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        assert dim % n_heads == 0
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.rotary = RotaryEmbedding(self.head_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        cos, sin = self.rotary(L)
        cos, sin = cos[None, None], sin[None, None]
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        attn = F.softmax((q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim), dim=-1)
        return self.proj((attn @ v).transpose(1, 2).reshape(B, L, D))


class TransformerLayer(nn.Module):
    """Single transformer layer"""
    def __init__(self, dim: int, n_heads: int = 8, mlp_ratio: int = 4,
                 use_attention: bool = True, seq_len: Optional[int] = None):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)
        if use_attention:
            self.mixer = SelfAttention(dim, n_heads)
        else:
            assert seq_len is not None, "seq_len required for MLP mixer"
            self.mixer = MLPMixer(seq_len, dim)
        self.mlp = SwiGLU(dim, int(dim * mlp_ratio))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.mixer(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Task Heads
# ---------------------------------------------------------------------------

class TaskHead(nn.Module):
    """Base class for task-specific input encoding and output decoding."""

    def encode(self, x_input: torch.Tensor) -> torch.Tensor:
        """Encode raw input tokens to embeddings [B, seq_len, dim]."""
        raise NotImplementedError

    def decode(self, hidden: torch.Tensor) -> torch.Tensor:
        """Decode normalized hidden state to task-specific logits."""
        raise NotImplementedError

    def compute_loss(self, logits: torch.Tensor, targets: torch.Tensor,
                     x_input: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute task-specific loss scalar."""
        raise NotImplementedError

    def check_correct(self, logits: torch.Tensor, targets: torch.Tensor,
                      x_input: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Return per-sample correctness as [B] float tensor."""
        raise NotImplementedError


class SudokuHead(TaskHead):
    """Task head for 9x9 Sudoku puzzles.

    Input:  [B, 81] integers 0-9  (0 = empty cell)
    Output: [B, 81, 9] logits over digits 1-9

    Loss is masked to only penalise originally-empty cells.
    Uses learnable 2D positional embeddings (row + column).
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.embedding = nn.Embedding(10, dim)
        self.row_embed = nn.Embedding(9, dim)
        self.col_embed = nn.Embedding(9, dim)
        self.output = nn.Linear(dim, 9, bias=False)
        self.register_buffer("_row_idx", torch.arange(9).repeat_interleave(9))
        self.register_buffer("_col_idx", torch.arange(9).repeat(9))

    def encode(self, x_input: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x_input)
        x = x + self.row_embed(self._row_idx)[None] + self.col_embed(self._col_idx)[None]
        return x

    def decode(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.output(hidden)

    def compute_loss(self, logits, targets, x_input=None):
        class_targets = targets - 1  # 1-9 -> 0-8
        if x_input is not None:
            mask = (x_input == 0)
            logits_flat = logits[mask]
            targets_flat = class_targets[mask]
            if targets_flat.numel() == 0:
                return torch.tensor(0.0, device=logits.device, requires_grad=True)
            return F.cross_entropy(logits_flat, targets_flat, reduction='mean')
        return F.cross_entropy(logits.reshape(-1, 9), class_targets.reshape(-1), reduction='mean')

    def check_correct(self, logits, targets, x_input=None):
        pred = logits.argmax(dim=-1) + 1
        if x_input is not None:
            mask = (x_input == 0)
            return ((pred == targets) | ~mask).all(dim=-1).float()
        return (pred == targets).all(dim=-1).float()


class MazeHead(TaskHead):
    """Task head for 2D maze solving.

    Input:  [B, H*W] integers 0-3 (0=wall, 1=open, 2=start, 3=goal)
    Output: [B, H*W, 1] logits (is this cell on the solution path?)

    Uses learnable 2D positional embeddings (row + column).
    """

    def __init__(self, dim: int, grid_size: int = 9):
        super().__init__()
        self.dim = dim
        self.grid_size = grid_size if grid_size % 2 == 1 else grid_size + 1
        self.embedding = nn.Embedding(4, dim)
        self.row_embed = nn.Embedding(self.grid_size, dim)
        self.col_embed = nn.Embedding(self.grid_size, dim)
        self.output = nn.Linear(dim, 1, bias=False)
        gs = self.grid_size
        self.register_buffer("_row_idx", torch.arange(gs).repeat_interleave(gs))
        self.register_buffer("_col_idx", torch.arange(gs).repeat(gs))

    def encode(self, x_input: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x_input)
        x = x + self.row_embed(self._row_idx)[None] + self.col_embed(self._col_idx)[None]
        return x

    def decode(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.output(hidden)

    def compute_loss(self, logits, targets, x_input=None):
        logits_flat = logits.squeeze(-1)
        pos_weight = torch.tensor([5.0], device=logits.device)
        return F.binary_cross_entropy_with_logits(
            logits_flat.reshape(-1),
            targets.float().reshape(-1),
            pos_weight=pos_weight.expand_as(logits_flat.reshape(-1)),
            reduction='mean',
        )

    def check_correct(self, logits, targets, x_input=None):
        pred = (logits.squeeze(-1) > 0).long()
        return (pred == targets).all(dim=-1).float()


# ---------------------------------------------------------------------------
# Core Recursive Model
# ---------------------------------------------------------------------------

class TinyRecursiveModel(nn.Module):
    """
    Tiny Recursive Model (TRM)

    The core recursive mechanism is task-agnostic. A TaskHead handles input
    encoding, output decoding, loss computation, and correctness checking.

    Args:
        task_head:     TaskHead instance (SudokuHead or MazeHead).
        dim:           Hidden dimension.
        n_layers:      Number of transformer layers (weight-shared).
        n_heads:       Number of attention heads.
        n_recursions:  Latent recursions per cycle (n).
        n_cycles:      Recursion cycles per supervision step (T).
        n_supervision: Maximum supervision / thinking steps.
        use_attention: Self-attention (True) or MLP-mixer (False).
        max_seq_len:   Required only for MLP-mixer mode.
    """

    def __init__(
        self,
        task_head: TaskHead,
        dim: int = 128,
        n_layers: int = 2,
        n_heads: int = 8,
        n_recursions: int = 8,
        n_cycles: int = 3,
        n_supervision: int = 14,
        use_attention: bool = True,
        max_seq_len: Optional[int] = None,
    ):
        super().__init__()
        self.dim = dim
        self.n_recursions = n_recursions
        self.n_cycles = n_cycles
        self.n_supervision = n_supervision
        self.task_head = task_head

        # Dynamic init vectors — broadcast to any seq_len at runtime
        self.y_init = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.z_init = nn.Parameter(torch.randn(1, 1, dim) * 0.02)

        self.layers = nn.ModuleList([
            TransformerLayer(dim, n_heads, use_attention=use_attention,
                             seq_len=max_seq_len if not use_attention else None)
            for _ in range(n_layers)
        ])

        self.output_norm = RMSNorm(dim)
        self.q_head = nn.Linear(dim, 1, bias=False)

    def forward_network(self, *inputs):
        """Apply the shared network to the sum of input tensors."""
        x = sum(inputs)
        for layer in self.layers:
            x = layer(x)
        return x

    def latent_recursion(self, x, y, z):
        """n latent recursions on z, then one y update."""
        for _ in range(self.n_recursions):
            z = self.forward_network(x, y, z)
        y = self.forward_network(y, z)
        return y, z

    def deep_recursion(self, x, y, z, with_gradients: bool = False):
        """T cycles of latent recursion. Returns (y, z), y_hat, q_hat."""
        if not with_gradients and self.n_cycles > 1:
            with torch.no_grad():
                for _ in range(self.n_cycles - 1):
                    y, z = self.latent_recursion(x, y, z)
        y, z = self.latent_recursion(x, y, z)
        y_norm = self.output_norm(y)
        y_hat = self.task_head.decode(y_norm)
        q_hat = torch.sigmoid(self.q_head(y_norm.mean(dim=1)))
        return (y, z), y_hat, q_hat

    def forward(
        self,
        x_input: torch.Tensor,
        y_true: Optional[torch.Tensor] = None,
        training: bool = True,
        return_all_steps: bool = False,
    ):
        """Forward pass with deep supervision.

        Args:
            x_input:          Raw input [B, seq_len].
            y_true:           Targets [B, ...] (training only).
            training:         Training vs inference mode.
            return_all_steps: Inference only — also return list of per-step preds.

        Returns:
            Training:  (losses, predictions, halts)
            Inference: y_hat  or  (y_hat, all_predictions)
        """
        B = x_input.shape[0]
        x = self.task_head.encode(x_input)
        seq_len = x.shape[1]

        y = self.y_init.expand(B, seq_len, -1)
        z = self.z_init.expand(B, seq_len, -1)

        if training:
            losses: List[torch.Tensor] = []
            predictions: List[torch.Tensor] = []
            halts: List[torch.Tensor] = []

            for _ in range(self.n_supervision):
                (y, z), y_hat, q_hat = self.deep_recursion(x, y, z, with_gradients=True)
                y, z = y.detach(), z.detach()
                predictions.append(y_hat)
                halts.append(q_hat)

                if y_true is not None:
                    pred_loss = self.task_head.compute_loss(y_hat, y_true, x_input)
                    is_correct = self.task_head.check_correct(y_hat, y_true, x_input)
                    halt_loss = F.binary_cross_entropy(q_hat.squeeze(-1), is_correct)
                    losses.append(pred_loss + 0.5 * halt_loss)

            return losses, predictions, halts
        else:
            all_preds: List[torch.Tensor] = []
            for _ in range(self.n_supervision):
                (y, z), y_hat, q_hat = self.deep_recursion(x, y, z, with_gradients=False)
                y, z = y.detach(), z.detach()
                all_preds.append(y_hat)

            return (y_hat, all_preds) if return_all_steps else y_hat


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_trm_model(
    task: str,
    dim: int = 128,
    n_layers: int = 2,
    n_heads: int = 8,
    n_recursions: int = 8,
    n_cycles: int = 3,
    n_supervision: int = 14,
    use_attention: bool = True,
    grid_size: int = 9,
    max_seq_len: Optional[int] = None,
    **kwargs,
) -> TinyRecursiveModel:
    """Create a TRM model for a given task.

    Args:
        task:      "sudoku" or "maze".
        dim:       Hidden dimension.
        grid_size: Grid size for maze (forced odd).
        **kwargs:  Extra args forwarded to TinyRecursiveModel.
    """
    if task == "sudoku":
        task_head = SudokuHead(dim)
        max_seq_len = max_seq_len or 81
    elif task == "maze":
        gs = grid_size if grid_size % 2 == 1 else grid_size + 1
        task_head = MazeHead(dim, grid_size=gs)
        max_seq_len = max_seq_len or gs * gs
    else:
        raise ValueError(f"Unknown task '{task}'. Choose 'sudoku' or 'maze'.")

    return TinyRecursiveModel(
        task_head=task_head,
        dim=dim,
        n_layers=n_layers,
        n_heads=n_heads,
        n_recursions=n_recursions,
        n_cycles=n_cycles,
        n_supervision=n_supervision,
        use_attention=use_attention,
        max_seq_len=max_seq_len,
        **kwargs,
    )
