import torch
from typing import Optional, Dict, Union
from .utils import YumaConfig


# Extracted from original file
def Yuma3(
    W: torch.Tensor,
    S: torch.Tensor,
    B_old: Optional[torch.Tensor] = None,
    config: YumaConfig = YumaConfig(),
    maxint: int = 2**64 - 1,
) -> Dict[str, Union[torch.Tensor | None, float]]:
    """
    Original Yuma function with bonds and EMA calculation.
    """

    # === Weight ===
    W = (W.T / (W.sum(dim=1) + 1e-6)).T

    # === Stake ===
    S = S / S.sum()

    # === Prerank ===
    P = (S.view(-1, 1) * W).sum(dim=0)

    # === Consensus ===
    C = torch.zeros(W.shape[1], dtype=torch.float64)

    for i, miner_weight in enumerate(W.T):
        c_high = 1.0
        c_low = 0.0

        while (c_high - c_low) > 1 / config.consensus_precision:
            c_mid = (c_high + c_low) / 2.0
            _c_sum = (miner_weight > c_mid) * S
            if _c_sum.sum() > config.kappa:
                c_low = c_mid
            else:
                c_high = c_mid

        C[i] = c_high

    C = (C / C.sum() * 65_535).int() / 65_535

    # === Consensus clipped weight ===
    W_clipped = torch.min(W, C)

    # === Rank ===
    R = (S.view(-1, 1) * W_clipped).sum(dim=0)

    # === Incentive ===
    I = (R / R.sum()).nan_to_num(0)

    # === Trusts ===
    T = (R / P).nan_to_num(0)
    T_v = W_clipped.sum(dim=1) / W.sum(dim=1)

    # === Bonds ===
    if B_old is None:
        B_old = torch.zeros_like(W)

    capacity = S * maxint

    # Compute Remaining Capacity
    capacity_per_bond = S.unsqueeze(1) * maxint
    remaining_capacity = capacity_per_bond - B_old
    remaining_capacity = torch.clamp(remaining_capacity, min=0.0)

    # Compute Purchase Capacity
    capacity_alpha = (config.capacity_alpha * capacity).unsqueeze(1)
    purchase_capacity = torch.min(capacity_alpha, remaining_capacity)

    # Allocate Purchase to Miners
    purchase = purchase_capacity * W

    # Update Bonds with Decay and Purchase
    decay = 1 - config.decay_rate
    B = decay * B_old + purchase
    B = torch.min(B, capacity_per_bond)  # Enforce capacity constraints

    # === Dividends Calculation ===
    D = (B * I).sum(dim=1)

    # Normalize dividends
    D_normalized = D / (D.sum() + 1e-6)

    return {
        "weight": W,
        "stake": S,
        "server_prerank": P,
        "server_consensus_weight": C,
        "consensus_clipped_weight": W_clipped,
        "server_rank": R,
        "server_incentive": I,
        "server_trust": T,
        "validator_trust": T_v,
        "validator_bonds": B,
        "validator_reward": D,
        "validator_reward_normalized": D_normalized,
    }
