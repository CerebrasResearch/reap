"""
Regression test for the fused_experts shape bug in MoETransformerObserver.

The original code at observer.py used:

    router_logits = module.router(flat_input)

which fails for routers that return a tuple (e.g. Llama4Router returns
(scores, logits)), and:

    .expand(router_scores.size(0), -1)

assuming router_scores has shape (num_experts, total_tokens).  In practice,
fused MoE models return router_scores with shape (total_tokens, num_experts),
so router_scores.size(0) is total_tokens, not num_experts.

This caused routed_in to be (total_tokens^2, hidden_dim) instead of
(num_experts * total_tokens, hidden_dim).  The .view(num_experts, ...) then
crashes because the element count doesn't match.

See: https://github.com/CerebrasResearch/reap/issues/11
"""

import torch
import torch.nn as nn
import pytest
from dataclasses import dataclass

from reap.observer import MoETransformerObserver, MoETransformerObserverConfig


# ---------------------------------------------------------------------------
# Minimal mocks that reproduce the same output contract as fused MoE modules:
# forward() returns (hidden_output, router_logits) where router_logits has
# shape (total_tokens, num_experts).
# ---------------------------------------------------------------------------

class MockFusedExperts(nn.Module):
    """Experts with a simple linear projection (fallback path)."""

    def __init__(self, hidden_dim):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.proj(x)


class MockFusedMoE(nn.Module):
    """Mimics fused MoE output: (output, router_logits)."""

    def __init__(self, num_experts, hidden_dim):
        super().__init__()
        self.router = nn.Linear(hidden_dim, num_experts)
        self.router.num_experts = num_experts
        self.router.top_k = 2
        self.experts = MockFusedExperts(hidden_dim)

    def forward(self, hidden_states):
        batch, seq, dim = hidden_states.shape
        flat = hidden_states.view(-1, dim)
        router_logits = self.router(flat)          # (total_tokens, num_experts)
        out = hidden_states                        # pass-through
        return out, router_logits


@dataclass
class MockFusedHookConfig(MoETransformerObserverConfig):
    module_class_name_to_hook_regex: str = "MockFusedMoE"
    num_experts_attr_name: str = "router.num_experts"
    top_k_attr_name: str = "router.top_k"
    fused_experts: bool = True


# ---------------------------------------------------------------------------


def test_fused_experts_fallback_works_when_tokens_ne_experts():
    """
    Before the fix, total_tokens=10 and num_experts=4 would crash with:

        routed_out.view(4, 10, 8)   # needs 320 elements
                                     # but routed_out had 10*10*8 = 800

    because router_scores.size(0) (total_tokens) was used instead of
    num_experts.  After the fix this should pass without error.
    """
    num_experts, hidden_dim = 4, 8
    batch_size, seq_len = 1, 10          # total_tokens=10 != num_experts=4

    moe = MockFusedMoE(num_experts, hidden_dim)
    model = nn.Sequential(moe)

    observer = MoETransformerObserver(
        model, hook_config=MockFusedHookConfig()
    )

    inp = torch.randn(batch_size, seq_len, hidden_dim)
    model(inp)  # should not raise

    state = observer.report_state()
    assert 0 in state, "observer should have recorded state for layer 0"
    assert state[0]["total_tokens"].item() == seq_len
