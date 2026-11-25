# -*- coding: utf-8 -*-
"""Matrix-free joint SVD primitives for direct-matrix LoKr + ordinary LoRA.

No Kronecker matrix or expanded full-rank LoKr A/B factors are materialized.
All operator inputs are expected to be CPU tensors for the merger's CPU workflow.
"""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class LoKrTerm:
    w1: object
    w2: object
    weight: float = 1.0


@dataclass(frozen=True)
class LoRATerm:
    A: object
    B: object
    scale: float = 1.0
    weight: float = 1.0


class MatrixFreeSum:
    """Linear operator for sum(weight * kron(w1,w2) + weight*scale*B@A)."""

    def __init__(self, torch, lokr_terms=(), lora_terms=()):
        self.torch = torch
        self.lokr_terms = tuple(lokr_terms)
        self.lora_terms = tuple(lora_terms)
        shapes = []
        for term in self.lokr_terms:
            if term.w1.ndim != 2 or term.w2.ndim != 2:
                raise ValueError("LoKr factors must be 2D matrices.")
            if min(*term.w1.shape, *term.w2.shape) < 1:
                raise ValueError("LoKr factors must be non-empty.")
            self._finite(term.weight, "LoKr weight")
            shapes.append((term.w1.shape[0] * term.w2.shape[0],
                           term.w1.shape[1] * term.w2.shape[1]))
        for term in self.lora_terms:
            if term.A.ndim != 2 or term.B.ndim != 2:
                raise ValueError("LoRA A/B factors must be 2D matrices.")
            if term.A.shape[0] != term.B.shape[1]:
                raise ValueError("LoRA A/B rank dimensions do not match.")
            if min(*term.A.shape, *term.B.shape) < 1:
                raise ValueError("LoRA factors must be non-empty.")
            self._finite(term.scale, "LoRA scale")
            self._finite(term.weight, "LoRA weight")
            shapes.append((term.B.shape[0], term.A.shape[1]))
        if not shapes:
            raise ValueError("At least one LoKr or LoRA term is required.")
        if any(s != shapes[0] for s in shapes[1:]):
            raise ValueError("All operator terms must have identical output/input dimensions.")
        self.shape = shapes[0]

    @staticmethod
    def _finite(value, name):
        if not math.isfinite(float(value)):
            raise ValueError(name + " must be finite.")

    def _check_rhs(self, x, expected_rows):
        if x.ndim != 2 or x.shape[0] != expected_rows:
            raise ValueError("Operator input has an incompatible shape.")
        if not self.torch.isfinite(x).all():
            raise ValueError("Operator input contains NaN or Inf.")

    def matmat(self, x):
        """Return M @ x, where x has shape (input_dim, columns)."""
        torch = self.torch
        self._check_rhs(x, self.shape[1])
        x = x.float()
        result = torch.zeros((self.shape[0], x.shape[1]), dtype=torch.float32, device=x.device)
        for term in self.lokr_terms:
            w1, w2 = term.w1.float(), term.w2.float()
            n1, n2 = w1.shape[1], w2.shape[1]
            block = x.reshape(n1, n2, -1)
            # Two small contractions; a 3-operand einsum may build the full
            # (m1, n1, m2, n2) Kronecker tensor when opt_einsum is absent.
            applied = torch.einsum("ij,jlc->ilc", w1, block)
            applied = torch.einsum("kl,ilc->ikc", w2, applied)
            result.add_(applied.reshape(self.shape[0], -1), alpha=float(term.weight))
        for term in self.lora_terms:
            update = term.B.float() @ (term.A.float() @ x)
            result.add_(update, alpha=float(term.weight) * float(term.scale))
        return result

    def rmatmat(self, x):
        """Return M.T @ x, where x has shape (output_dim, columns)."""
        torch = self.torch
        self._check_rhs(x, self.shape[0])
        x = x.float()
        result = torch.zeros((self.shape[1], x.shape[1]), dtype=torch.float32, device=x.device)
        for term in self.lokr_terms:
            w1, w2 = term.w1.float(), term.w2.float()
            m1, m2 = w1.shape[0], w2.shape[0]
            block = x.reshape(m1, m2, -1)
            applied = torch.einsum("ij,jlc->ilc", w1.T, block)
            applied = torch.einsum("kl,ilc->ikc", w2.T, applied)
            result.add_(applied.reshape(self.shape[1], -1), alpha=float(term.weight))
        for term in self.lora_terms:
            update = term.A.float().T @ (term.B.float().T @ x)
            result.add_(update, alpha=float(term.weight) * float(term.scale))
        return result


def randomized_joint_svd(operator, rank, oversample=8, power_iterations=2,
                         seed=0, error_probes=4):
    """Approximate M by B @ A; residual is estimated with independent probes.

    Returns (A, B, relative_residual_estimate, projected_rank). The output
    factors are zero-padded to exactly the requested rank.

    The estimate is ||(M - B@A) G||_F / ||M G||_F for a Gaussian probe block G
    independent of the sketch. Since E||X G||_F^2 = p ||X||_F^2, it estimates the
    relative Frobenius error of the FINAL factors against the TRUE operator M
    (randomization error + rank truncation together). It is a noisy estimate
    (std ~10% of its value at 8 probes in the synthetic tests), not a bound and
    not the Eckart-Young optimum.
    """
    torch = operator.torch
    m, n = operator.shape
    rank = int(rank)
    oversample = int(oversample)
    power_iterations = int(power_iterations)
    error_probes = int(error_probes)
    if rank < 1 or oversample < 0 or power_iterations < 0 or error_probes < 1:
        raise ValueError("rank/error_probes must be positive; oversample and power_iterations non-negative.")
    width = min(min(m, n), rank + oversample)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    omega = torch.randn((n, width), generator=generator, dtype=torch.float32)
    y = operator.matmat(omega)
    q, _ = torch.linalg.qr(y, mode="reduced")
    for _ in range(power_iterations):
        z = operator.rmatmat(q)
        z, _ = torch.linalg.qr(z, mode="reduced")
        y = operator.matmat(z)
        q, _ = torch.linalg.qr(y, mode="reduced")
    c = operator.rmatmat(q).T.contiguous()
    uc, singular, vh = torch.linalg.svd(c, full_matrices=False)
    k = min(rank, int(singular.numel()))
    root = torch.sqrt(torch.clamp(singular[:k], min=0))
    b_small = (q @ uc[:, :k]) * root.unsqueeze(0)
    a_small = root.unsqueeze(1) * vh[:k, :]
    A = torch.zeros((rank, n), dtype=torch.float32)
    B = torch.zeros((m, rank), dtype=torch.float32)
    if k:
        A[:k, :] = a_small
        B[:, :k] = b_small
    probes = torch.randn((n, error_probes), generator=generator, dtype=torch.float32)
    actual = operator.matmat(probes)
    approx = B @ (A @ probes)
    denom = torch.linalg.vector_norm(actual)
    residual = torch.linalg.vector_norm(actual - approx)
    relative = 0.0 if float(denom.item()) == 0.0 else float((residual / denom).item())
    return A.contiguous(), B.contiguous(), relative, k


def estimate_workspace_bytes(shape, rank, oversample=8, dtype_bytes=4,
                             safety_factor=2.0, error_probes=8):
    """Heuristic algorithm-workspace estimate; excludes source payloads/runtime.

    Not a guarantee that the process fits in RAM: transient copies (float32
    casts, einsum intermediates, SVD/QR scratch) are covered only by safety_factor.
    """
    m, n = map(int, shape)
    rank, oversample, dtype_bytes = int(rank), int(oversample), int(dtype_bytes)
    if min(m, n, rank, dtype_bytes) < 1 or oversample < 0:
        raise ValueError("Invalid dimensions or workspace parameters.")
    width = min(min(m, n), rank + oversample)
    elements = m * width + n * width + rank * (m + n) + (n + 2 * m) * int(error_probes)
    return int(math.ceil(elements * dtype_bytes * float(safety_factor)))
