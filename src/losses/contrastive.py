from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SymmetricInfoNCELoss(nn.Module):
    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        sample_weights: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if queries.ndim != 2 or keys.ndim != 2:
            raise ValueError("queries and keys must be rank-2 tensors")
        if queries.shape != keys.shape:
            raise ValueError("queries and keys must have the same shape")

        logits = queries @ keys.t()
        logits = logits / self.temperature
        labels = torch.arange(logits.shape[0], device=logits.device)

        q_to_k_per_sample = F.cross_entropy(logits, labels, reduction="none")
        k_to_q_per_sample = F.cross_entropy(logits.t(), labels, reduction="none")

        if sample_weights is not None:
            if sample_weights.ndim != 1 or sample_weights.shape[0] != logits.shape[0]:
                raise ValueError("sample_weights must be a rank-1 tensor aligned with the batch dimension")
            normalized_weights = sample_weights.to(device=logits.device, dtype=logits.dtype)
            normalized_weights = normalized_weights / normalized_weights.mean().clamp_min(1e-12)
            q_to_k = (q_to_k_per_sample * normalized_weights).mean()
            k_to_q = (k_to_q_per_sample * normalized_weights).mean()
        else:
            q_to_k = q_to_k_per_sample.mean()
            k_to_q = k_to_q_per_sample.mean()

        loss = 0.5 * (q_to_k + k_to_q)

        with torch.no_grad():
            top1 = (logits.argmax(dim=1) == labels).float().mean()

        return {
            "loss": loss,
            "loss_q_to_k": q_to_k.detach(),
            "loss_k_to_q": k_to_q.detach(),
            "retrieval_acc": top1.detach(),
        }


__all__ = ["SymmetricInfoNCELoss"]
