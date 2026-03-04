"""
models/random_field.py

T-partite Random Field with Belief Propagation for trajectory generation.

Implements:
  • Frame-level potentials       
  • Visual relationship sim.     
  • Full objective               
  • Belief propagation solver    
  • Greedy solver                
  • Frame-level-only solver      
"""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Potential computations
# ─────────────────────────────────────────────────────────────────────────────

def frame_level_potential(
    rel_emb_test:    torch.Tensor,   # (T, M, M, D) – test video embeddings
    rel_emb_support: List[torch.Tensor],  # K × (T_k, M, M, D) – support embeddings
    rn,                               # RelationNetwork instance
) -> torch.Tensor:                   # (T, M, M)
    """
    Ψ_i(u_i^{jk}, r, θ) = - (1/K) Σ_m R_θ(f_r(u_i^{jk}), f_r(u_{im}^{jk}))
    (Eq. 2)

    For each support video we average over its frames to obtain a single
    representative embedding per (j, k) pair.
    """
    T, M, _, D = rel_emb_test.shape
    K = len(rel_emb_support)

    # Per-support-video representative: mean over time dimension → (M, M, D)
    sup_means = [s.mean(dim=0) for s in rel_emb_support]  # K × (M, M, D)
    sup_stack = torch.stack(sup_means, dim=0)              # (K, M, M, D)

    # Expand test embeddings: (T, 1, M, M, D) and support: (1, K, M, M, D)
    test_exp = rel_emb_test.unsqueeze(1)         # (T, 1, M, M, D)
    sup_exp  = sup_stack.unsqueeze(0)            # (1, K, M, M, D)

    test_exp = test_exp.expand(-1, K, -1, -1, -1)  # (T, K, M, M, D)
    sup_exp  = sup_exp.expand(T, -1, -1, -1, -1)   # (T, K, M, M, D)

    scores = rn(test_exp, sup_exp)               # (T, K, M, M)
    psi_i  = -scores.mean(dim=1)                 # (T, M, M)  – negative avg sim
    return psi_i


def visual_relationship_similarity_potential(
    rel_emb: torch.Tensor,   # (T, M, M, D)
    rn,                       # RelationNetwork instance
) -> torch.Tensor:            # (T-1, M, M)
    """
    Ψ_il(u_i^{jk}, u_l^{jk}, θ) = -R_θ(f_r(u_i^{jk}), f_r(u_l^{jk}))
    for l = i+1 (consecutive frames, Eq. 3).
    """
    T, M, _, D = rel_emb.shape
    f_cur  = rel_emb[:-1]   # (T-1, M, M, D)
    f_next = rel_emb[1:]    # (T-1, M, M, D)
    psi_il = -rn(f_cur, f_next)  # (T-1, M, M)
    return psi_il


# ─────────────────────────────────────────────────────────────────────────────
# Solvers
# ─────────────────────────────────────────────────────────────────────────────

SKIP_THRESHOLD = -0.5


def solve_frame_level(
    psi_i: torch.Tensor,   # (T, M, M)
    threshold: float = SKIP_THRESHOLD,
) -> Tuple[List[int], List[int]]:
    """
    Greedy frame-by-frame selection using only Ψ_i.
    Returns subject and object trajectory as lists of box indices (or -1 to skip).
    """
    T, M, _ = psi_i.shape
    subj_traj, obj_traj = [], []
    for t in range(T):
        min_val = psi_i[t].min()
        if min_val.item() > threshold:
            subj_traj.append(-1)
            obj_traj.append(-1)
        else:
            flat_idx = psi_i[t].argmin()
            j = (flat_idx // M).item()
            k = (flat_idx %  M).item()
            subj_traj.append(j)
            obj_traj.append(k)
    return subj_traj, obj_traj


def solve_greedy(
    psi_i:  torch.Tensor,    # (T, M, M)
    psi_il: torch.Tensor,    # (T-1, M, M)
    threshold: float = SKIP_THRESHOLD,
) -> Tuple[List[int], List[int]]:
    """
    Greedy solver: select the best node at frame 0 using Ψ_i, then
    for each subsequent frame pick the transition with minimal Ψ_il.
    """
    T, M, _ = psi_i.shape
    subj_traj, obj_traj = [], []

    # Frame 0
    min_val = psi_i[0].min()
    if min_val.item() > threshold:
        subj_traj.append(-1); obj_traj.append(-1)
        prev_j = prev_k = 0
    else:
        flat_idx = psi_i[0].argmin()
        prev_j = (flat_idx // M).item()
        prev_k = (flat_idx %  M).item()
        subj_traj.append(prev_j); obj_traj.append(prev_k)

    for t in range(1, T):
        trans_cost = psi_il[t - 1]  # (M, M) – cost of (j, k) at frame t
        min_val = trans_cost.min()
        if min_val.item() > threshold:
            subj_traj.append(-1); obj_traj.append(-1)
        else:
            flat_idx = trans_cost.argmin()
            j = (flat_idx // M).item()
            k = (flat_idx %  M).item()
            subj_traj.append(j); obj_traj.append(k)
            prev_j, prev_k = j, k

    return subj_traj, obj_traj


def solve_belief_propagation(
    psi_i:       torch.Tensor,   # (T, M, M)
    psi_il:      torch.Tensor,   # (T-1, M, M)
    iterations:  int   = 10,
    threshold:   float = SKIP_THRESHOLD,
) -> Tuple[List[int], List[int]]:
    """
    Loopy belief propagation on the T-partite chain graph.

    Each node u_i^{jk} is a binary variable {select, reject}.
    We keep only the marginal for select=1 and compute MAP via max-product BP.

    For a chain graph (left-to-right and right-to-left passes):
      μ_{i→i+1}(u) = min_{v} [Ψ_i(v) + Ψ_il(v, u) + μ_{i-1→i}(v)]
    We run 'iterations' forward-backward sweeps.
    """
    T, M, _ = psi_i.shape
    N = M * M  # number of nodes per frame

    # Reshape to (T, N)
    unary  = psi_i.view(T, N)               # (T, N)
    pairwise = psi_il.view(T - 1, N, N) if psi_il.numel() > 0 else None

    # For a chain we use min-sum (log domain max-product)
    # Messages: msg[i] = message from frame i to frame i+1, shape (N,)
    msg_fwd = torch.zeros(T, N, device=psi_i.device)   # forward messages
    msg_bwd = torch.zeros(T, N, device=psi_i.device)   # backward messages

    for _ in range(iterations):
        # Forward pass
        for t in range(1, T):
            if pairwise is not None:
                # msg_fwd[t, u] = min_v [unary[t-1, v] + pairwise[t-1, v, u] + msg_fwd[t-1, v]]
                belief_prev = unary[t - 1] + msg_fwd[t - 1]  # (N,)
                # (N_v, 1) + (N_v, N_u) → (N_v, N_u) → min over v → (N_u,)
                cost = belief_prev.unsqueeze(1) + pairwise[t - 1]
                msg_fwd[t] = cost.min(dim=0).values
            else:
                msg_fwd[t] = unary[t - 1] + msg_fwd[t - 1]

        # Backward pass
        for t in range(T - 2, -1, -1):
            if pairwise is not None:
                belief_next = unary[t + 1] + msg_bwd[t + 1]  # (N,)
                cost = belief_next.unsqueeze(0) + pairwise[t]  # (N_u, N_v) → min v
                msg_bwd[t] = cost.min(dim=1).values
            else:
                msg_bwd[t] = unary[t + 1] + msg_bwd[t + 1]

    # MAP: argmin of total belief
    belief = unary + msg_fwd + msg_bwd  # (T, N)

    subj_traj, obj_traj = [], []
    for t in range(T):
        min_val = belief[t].min()
        if min_val.item() > threshold:
            subj_traj.append(-1); obj_traj.append(-1)
        else:
            flat_idx = belief[t].argmin().item()
            j = flat_idx // M
            k = flat_idx %  M
            subj_traj.append(j); obj_traj.append(k)

    return subj_traj, obj_traj


# ─────────────────────────────────────────────────────────────────────────────
# High-level wrapper
# ─────────────────────────────────────────────────────────────────────────────

def generate_trajectories(
    rel_emb_test:    torch.Tensor,
    rel_emb_support: List[torch.Tensor],
    rn,
    solver: str = "belief_propagation",
    bp_iterations: int = 10,
    threshold: float = SKIP_THRESHOLD,
) -> Tuple[List[int], List[int]]:
    """
    End-to-end: compute potentials and solve for trajectories.

    Args:
        rel_emb_test    : (T, M, M, D)
        rel_emb_support : list of K tensors, each (T_k, M, M, D)
        rn              : RelationNetwork instance
        solver          : "frame_level" | "greedy" | "belief_propagation"
    Returns:
        subj_traj, obj_traj: list of box indices per frame (-1 = skip)
    """
    with torch.no_grad():
        psi_i  = frame_level_potential(rel_emb_test, rel_emb_support, rn)
        psi_il = visual_relationship_similarity_potential(rel_emb_test, rn)

    if solver == "frame_level":
        return solve_frame_level(psi_i, threshold)
    elif solver == "greedy":
        return solve_greedy(psi_i, psi_il, threshold)
    else:  # belief_propagation (default)
        return solve_belief_propagation(psi_i, psi_il, bp_iterations, threshold)
