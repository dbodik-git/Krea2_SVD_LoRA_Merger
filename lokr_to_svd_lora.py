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


def estimate_full_rank_factor_bytes(tensors):
    """Estimate float32 A/B storage needed for an exact Kronecker factorization."""
    groups = _group_lokr(tensors)
    total = 0
    for stem, pair in groups.items():
        s1, s2 = pair["w1"]["shape"], pair["w2"]["shape"]
        if len(s1) != 2 or len(s2) != 2 or min(s1 + s2) < 1:
            raise ValueError(f"LoKr factors must be non-empty 2D matrices at {stem}.")
        rank = min(s1) * min(s2)
        out_dim, in_dim = s1[0] * s2[0], s1[1] * s2[1]
        total += 4 * rank * (out_dim + in_dim)
    return total


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
            "has_alpha": "alpha" in pair,
        }
        dtypes.update((w1["dtype"], w2["dtype"]))
    notes = [
        "Detected full-matrix LoKr; it will be converted to ordinary SVD LoRA "
        "when the merge runs (rank depends on selected conversion mode)."
    ]
    if not metadata.get("modelspec.architecture"):
        notes.append("Metadata has no modelspec.architecture.")
    return {
        "path": path, "metadata": metadata, "modules": modules,
        "tensor_count": len(tensors), "module_count": len(modules),
        "ranks": [], "dtypes": sorted(dtypes), "warnings": [], "notes": notes,
        "format": "lokr",
        "lokr_full_rank_factor_bytes": estimate_full_rank_factor_bytes(tensors),
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
    if target_rank is None:
        target_rank = int(flat.numel())
    k = min(int(target_rank), int(flat.numel()))
    # Full sort (N = min(m1,n1)*min(m2,n2), small) so the discarded tail is known
    # explicitly and the error is a sum of the dropped squares, not total - kept.
    sorted_values, sorted_indices = torch.sort(flat, descending=True)
    values, indices = sorted_values[:k], sorted_indices[:k]
    tail_values = sorted_values[k:]
    i1 = torch.div(indices, s2.numel(), rounding_mode="floor")
    i2 = indices.remainder(s2.numel())
    roots = torch.sqrt(torch.clamp(values, min=0))

    out_dim = w1.shape[0] * w2.shape[0]
    in_dim = w1.shape[1] * w2.shape[1]
    B = torch.zeros((out_dim, target_rank), dtype=torch.float32, device="cpu")
    A = torch.zeros((target_rank, in_dim), dtype=torch.float32, device="cpu")
    # Column c of kron(U1, U2) is kron(u1[:, i1[c]], u2[:, i2[c]]). Filled in column
    # chunks so the temporaries stay a small fraction of B/A (peak ~1x, not ~3x).
    u1s, u2s = u1[:, i1], u2[:, i2]            # (m1, k), (m2, k)
    v1s, v2s = vh1[i1, :].T, vh2[i2, :].T      # (n1, k), (n2, k)
    chunk = max(1, min(k, (1 << 22) // max(1, max(out_dim, in_dim))))
    for c0 in range(0, k, chunk):
        c1 = min(k, c0 + chunk)
        r = roots[c0:c1].unsqueeze(0)
        left = (u1s[:, None, c0:c1] * u2s[None, :, c0:c1]).reshape(out_dim, c1 - c0)
        B[:, c0:c1] = left * r
        right = (v1s[:, None, c0:c1] * v2s[None, :, c0:c1]).reshape(in_dim, c1 - c0)
        A[c0:c1, :] = (right * r).T
        del left, right
    # Frobenius norm of kron(w1, w2) is sqrt(sum(flat**2)), so the relative error of
    # the rank-k truncation is known exactly without building the full matrix.
    total_energy = float((flat.double() ** 2).sum().item())
    tail_energy = float((tail_values.double() ** 2).sum().item())   # no cancellation
    rel_error = 0.0 if total_energy <= 0 else math.sqrt(tail_energy / total_energy)
    return B.contiguous(), A.contiguous(), k, int(flat.numel()), rel_error


LOKR_ALPHA_STATUS = "full-matrix LoKr applied as kron(w1, w2) with scale 1; alpha ignored; UNVERIFIED for this file's trainer"


def lokr_semantics_note(source_name, modules_with_alpha, total_modules):
    """One-time diagnostic stating what is and is not established about LoKr alpha.

    Established (secondary evidence, see README): LyCORIS-derived code sets scale = 1 when both
    lokr_w1 and lokr_w2 are full matrices. NOT established: that THIS file was written by such a trainer.
    A file without alpha tensors is typical of PEFT-style LoKr, which computes scaling = alpha/r for
    its layers; whether that is applied to full-matrix factors was not verified, so the true scale
    cannot be determined from the file alone.
    """
    if modules_with_alpha == 0:
        origin = ("it stores NO alpha tensors (typical of PEFT-style files, whose alpha lives in "
                  "adapter_config.json), so the producer's scale cannot be determined from the file")
    elif modules_with_alpha < total_modules:
        origin = f"only {modules_with_alpha}/{total_modules} modules store alpha (mixed or edited file)"
    else:
        origin = "it stores alpha tensors (consistent with LyCORIS-style files)"
    return (f"  NOTE (LoKr alpha, UNVERIFIED): {source_name}: {LOKR_ALPHA_STATUS}. Here {origin}. "
            "If the producer scaled full-matrix LoKr by alpha/rank, compensate with the merge coefficient "
            "(coefficient *= alpha/rank). See README, section 'LoKr alpha semantics'.")


def lokr_alpha_message(source_name, alpha_values, metadata=None):
    """Warning text if LoKr alpha values cannot be explained by the LyCORIS convention.

    For full-matrix LoKr, LyCORIS saves alpha equal to the configured dim and applies
    scale 1, so alpha == ss_network_dim is the expected, silent case.
    """
    values = sorted({round(float(v), 6) for v in alpha_values})
    if not values:
        return None
    dim = None
    if metadata:
        try:
            dim = float(metadata.get("ss_network_dim"))
        except (TypeError, ValueError):
            dim = None
    if dim is not None and all(abs(v - dim) < 1e-6 for v in values):
        return None
    shown = ", ".join(f"{v:g}" for v in values[:4]) + (", ..." if len(values) > 4 else "")
    ref = (f"metadata ss_network_dim={dim:g}" if dim is not None
           else "no ss_network_dim metadata to compare with")
    return (f"WARNING: {source_name} stores LoKr alpha values ({shown}; {ref}). Alpha is ignored: "
            "full-matrix LoKr is applied as kron(w1, w2) with scale 1 (LyCORIS convention). "
            "If this file came from a trainer that scales full LoKr by alpha/rank, its contribution "
            "will be off by that factor; compensate with the merge weight.")


def convert_lokr_state_dict(state_dict, torch, target_rank, source_name="<LoKr>", metadata=None,
                            stage1_report=None):
    """Convert direct-matrix LoKr tensors to standard Kohya-style LoRA tensors.

    With an integer target_rank, the update is truncated during conversion.
    With target_rank=None, the exact Kronecker-product rank is preserved.
    Alpha is emitted as the resulting factor rank, giving the ordinary merger
    an effective scale of exactly one.

    If ``stage1_report`` is a dict, it is filled with
    ``{stem: absolute Frobenius truncation error of this LoKr module}`` so the
    caller can bound the combined (convert + merge) error.
    """
    if target_rank is not None and int(target_rank) < 1:
        raise ValueError("LoKr conversion target rank must be >= 1.")
    groups = _group_lokr(state_dict)
    converted = {}
    alpha_values, trunc_errors = [], []
    total_modules = len(groups)
    mode = "exact full rank (no intermediate truncation)" if target_rank is None else f"rank {target_rank}"
    print(
        f"  Detected LoKr in {source_name}: {total_modules} modules; "
        f"converting to ordinary SVD LoRA at {mode}."
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
            alpha_values.append(float(alpha[0].item()))
        B, A, _, _, rel_error = _kronecker_svd(w1, w2, target_rank, torch, stem)
        trunc_errors.append((rel_error, stem))
        if stage1_report is not None:
            # ||kron(w1, w2)||_F = ||w1||_F * ||w2||_F
            full_norm = float(torch.linalg.vector_norm(w1.float().reshape(-1)).item()
                              * torch.linalg.vector_norm(w2.float().reshape(-1)).item())
            stage1_report[stem] = rel_error * full_norm
        converted[stem + ".lora_down.weight"] = A
        converted[stem + ".lora_up.weight"] = B
        converted[stem + ".alpha"] = torch.tensor(float(A.shape[0]), dtype=torch.float32)
        if index % 25 == 0 or index == total_modules:
            print(f"    converted {index}/{total_modules} LoKr modules")
    print(lokr_semantics_note(source_name, len(alpha_values), total_modules))
    message = lokr_alpha_message(source_name, alpha_values, metadata)
    if message:
        print("  " + message)
    if target_rank is not None and trunc_errors:
        mean_err = sum(e for e, _ in trunc_errors) / len(trunc_errors)
        worst_err, worst_stem = max(trunc_errors)
        print(f"  LoKr -> rank {target_rank} truncation (applied BEFORE merging): "
              f"relative error mean={mean_err:.4g}, max={worst_err:.4g} ({worst_stem})")
        if worst_err > 0.01:
            # 0.01 is only a diagnostic hint, not a quality criterion.
            print("  NOTE: this loss is on top of the final merge truncation (the merge step's own "
                  "error figure does not include it). Use --lokr-full-rank or --lokr-matrix-free "
                  "to truncate only once.")
    return converted
