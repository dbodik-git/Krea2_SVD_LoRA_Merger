# -*- coding: utf-8 -*-
"""Convert full-matrix LyCORIS LoKr modules into ordinary SVD LoRA factors.

This module deliberately handles the direct lokr_w1/lokr_w2 layout used by
Krea2 LoKr files. It never materializes torch.kron(w1, w2): the SVD of a
Kronecker product is assembled from the SVDs of its two small factors.
"""
from collections import defaultdict
import math


_W1 = ".lokr_w1"
_W2 = ".lokr_w2"
_ALPHA = ".alpha"
_UNSUPPORTED = (
    ".lokr_w1_a", ".lokr_w1_b", ".lokr_w2_a", ".lokr_w2_b", ".lokr_t2",
)


def is_lokr_state_keys(keys):
    """Return True when any tensor key identifies a LoKr layout."""
    suffixes = (_W1, _W2) + _UNSUPPORTED
    return any(isinstance(key, str) and key.endswith(suffixes) for key in keys)


def _group_lokr(items):
    groups = defaultdict(dict)
    unknown = []
    for key, value in items.items():
        if key == "__metadata__":
            continue
        matched = False
        for suffix, kind in ((_W1, "w1"), (_W2, "w2"), (_ALPHA, "alpha")):
            if key.endswith(suffix):
                stem = key[:-len(suffix)]
                if kind in groups[stem]:
                    raise ValueError(f"Duplicate LoKr {kind} tensor: {stem}")
                groups[stem][kind] = value
                matched = True
                break
        if not matched:
            unknown.append(key)
    if unknown:
        if any(key.endswith(_UNSUPPORTED) for key in unknown):
            raise ValueError(
                "This LoKr variant uses decomposed/Tucker factors "
                "(lokr_w*_a/b or lokr_t2); only direct lokr_w1/lokr_w2 "
                "matrices are currently supported. Unsupported keys: "
                + ", ".join(unknown[:8])
            )
        raise ValueError(
            "LoKr file contains unrecognized or mixed tensor keys: "
            + ", ".join(unknown[:8])
        )
    if not groups:
        raise ValueError("No LoKr lokr_w1/lokr_w2 modules were found.")
    for stem, pair in groups.items():
        if "w1" not in pair or "w2" not in pair:
            raise ValueError(f"Incomplete LoKr pair at {stem}: expected lokr_w1 and lokr_w2.")
    return groups


def inspect_lokr_header(path, metadata, tensors):
    """Build LoRA-compatible synthetic dimensions from a LoKr SafeTensors header."""
    groups = _group_lokr(tensors)
    modules, dtypes = {}, set()
    for stem, pair in groups.items():
        w1 = pair["w1"]
        w2 = pair["w2"]
        s1, s2 = w1["shape"], w2["shape"]
        if len(s1) != 2 or len(s2) != 2 or min(s1 + s2) < 1:
            raise ValueError(
                f"LoKr factors must be non-empty 2D matrices at {stem}: "
                f"w1={s1}, w2={s2}"
            )
        if "alpha" in pair:
            ashape = pair["alpha"]["shape"]
            if len(ashape) > 1 or (math.prod(ashape) if ashape else 1) != 1:
                raise ValueError(f"LoKr alpha must be scalar at {stem}: {ashape}")
        out_dim, in_dim = s1[0] * s2[0], s1[1] * s2[1]
        # The rank-one placeholder is used only for input-dimension compatibility
        # checks. The real factors are created at the selected target rank at run time.
        modules[stem] = {
            "A": {"shape": (1, in_dim), "dtype": w2["dtype"]},
            "B": {"shape": (out_dim, 1), "dtype": w2["dtype"]},
            "A_suffix": ".lora_down.weight",
            "B_suffix": ".lora_up.weight",
        }
        dtypes.update((w1["dtype"], w2["dtype"]))
    notes = [
        "Detected full-matrix LoKr; it will be converted to ordinary SVD LoRA "
        "at the selected target rank when the merge runs."
    ]
    if not metadata.get("modelspec.architecture"):
        notes.append("Metadata has no modelspec.architecture.")
    return {
        "path": path, "metadata": metadata, "modules": modules,
        "tensor_count": len(tensors), "module_count": len(modules),
        "ranks": [], "dtypes": sorted(dtypes), "warnings": [], "notes": notes,
        "format": "lokr",
    }


def _kronecker_svd(w1, w2, target_rank, torch, stem):
    """Return B, A for the best rank-k approximation to kron(w1, w2)."""
    if w1.ndim != 2 or w2.ndim != 2:
        raise ValueError(
            f"LoKr factors must be 2D at {stem}: "
            f"w1={tuple(w1.shape)}, w2={tuple(w2.shape)}"
        )
    if min(*w1.shape, *w2.shape) < 1:
        raise ValueError(f"Empty LoKr factor at {stem}.")
    # LyCORIS stores full Kronecker factors without alpha/rank scaling.
    # Alpha is only a scale for decomposed LoKr factors; those are rejected above.
    w1f, w2f = w1.float(), w2.float()
    u1, s1, vh1 = torch.linalg.svd(w1f, full_matrices=False)
    u2, s2, vh2 = torch.linalg.svd(w2f, full_matrices=False)

    products = s1[:, None] * s2[None, :]
    flat = products.reshape(-1)
    k = min(int(target_rank), int(flat.numel()))
    values, indices = torch.topk(flat, k=k, largest=True, sorted=True)
    i1 = torch.div(indices, s2.numel(), rounding_mode="floor")
    i2 = indices.remainder(s2.numel())
    roots = torch.sqrt(torch.clamp(values, min=0))

    out_dim = w1.shape[0] * w2.shape[0]
    in_dim = w1.shape[1] * w2.shape[1]
    B = torch.zeros((out_dim, target_rank), dtype=torch.float32, device="cpu")
    A = torch.zeros((target_rank, in_dim), dtype=torch.float32, device="cpu")
    for col in range(k):
        left = torch.outer(u1[:, i1[col]], u2[:, i2[col]]).reshape(-1)
        right = torch.outer(vh1[i1[col], :], vh2[i2[col], :]).reshape(-1)
        B[:, col] = left * roots[col]
        A[col, :] = right * roots[col]
    return B.contiguous(), A.contiguous(), k, int(flat.numel())


def convert_lokr_state_dict(state_dict, torch, target_rank, source_name="<LoKr>"):
    """Convert direct-matrix LoKr tensors to standard Kohya-style LoRA tensors.

    The resulting update is the best rank-target_rank approximation to
    kron(lokr_w1, lokr_w2). Alpha is emitted as target_rank, so the ordinary
    merger applies an effective scale of exactly one.
    """
    if int(target_rank) < 1:
        raise ValueError("LoKr conversion target rank must be >= 1.")
    groups = _group_lokr(state_dict)
    converted = {}
    total_modules = len(groups)
    print(
        f"  Detected LoKr in {source_name}: {total_modules} modules; "
        f"converting to ordinary SVD LoRA at rank {target_rank}."
    )
    for index, (stem, pair) in enumerate(sorted(groups.items()), 1):
        w1, w2 = pair["w1"], pair["w2"]
        if w1.ndim != 2 or w2.ndim != 2:
            raise ValueError(
                f"{source_name}: LoKr factors must be 2D at {stem}: "
                f"w1={tuple(w1.shape)}, w2={tuple(w2.shape)}"
            )
        if "alpha" in pair:
            alpha = pair["alpha"].float().reshape(-1)
            if alpha.numel() != 1 or not torch.isfinite(alpha).all():
                raise ValueError(f"{source_name}: invalid LoKr alpha at {stem}.")
        B, A, _, _ = _kronecker_svd(w1, w2, int(target_rank), torch, stem)
        converted[stem + ".lora_down.weight"] = A
        converted[stem + ".lora_up.weight"] = B
        converted[stem + ".alpha"] = torch.tensor(float(target_rank), dtype=torch.float32)
        if index % 25 == 0 or index == total_modules:
            print(f"    converted {index}/{total_modules} LoKr modules")
    return converted
