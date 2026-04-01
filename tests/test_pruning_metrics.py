import torch

from reap.pruning_metrics import initialize_pruning_state, update_pruning_state


def test_update_pruning_state_filters_masked_tokens():
    layer_state = initialize_pruning_state(2)

    activations = torch.tensor(
        [
            [[3.0, 4.0], [1.0, 0.0], [5.0, 12.0]],
            [[0.0, 2.0], [0.0, 4.0], [8.0, 15.0]],
        ]
    )
    selected_experts = torch.tensor([[0], [1], [0]], dtype=torch.long)
    router_logits = torch.tensor(
        [[2.0, 1.0], [0.0, 3.0], [4.0, 0.0]], dtype=torch.float32
    )
    valid_token_mask = torch.tensor([True, False, True])

    pruning_batch = update_pruning_state(
        layer_state,
        activations=activations,
        selected_experts=selected_experts,
        router_logits=router_logits,
        num_experts=2,
        valid_token_mask=valid_token_mask,
    )

    expected_router_logits = router_logits[valid_token_mask]
    expected_routing_weights = torch.softmax(expected_router_logits, dim=1)
    expected_weighted_freq = expected_routing_weights[:, 0].sum().to(torch.float64)
    expected_weighted_ean_sum = (
        torch.tensor([5.0, 13.0]) * expected_routing_weights[:, 0]
    ).sum().to(torch.float64)
    expected_reap = (
        torch.tensor([5.0, 13.0]) * expected_routing_weights[:, 0]
    ).mean()

    assert pruning_batch.activations.shape == (2, 2, 2)
    assert torch.equal(pruning_batch.selected_experts, torch.tensor([[0], [0]]))
    assert torch.equal(pruning_batch.router_logits, expected_router_logits)
    assert pruning_batch.num_tokens.item() == 2
    assert torch.equal(pruning_batch.expert_frequency, torch.tensor([2, 0]))
    assert torch.equal(
        pruning_batch.pairwise_expert_frequency,
        torch.tensor([[4, 2], [2, 0]]),
    )

    assert layer_state["total_tokens"].item() == 2
    assert torch.equal(layer_state["expert_frequency"], torch.tensor([2, 0]))
    assert torch.equal(
        layer_state["pairwise_expert_frequency"],
        torch.tensor([[4, 2], [2, 0]]),
    )
    assert torch.allclose(
        layer_state["ean_sum"], torch.tensor([18.0, 0.0], dtype=torch.float64)
    )
    assert torch.allclose(
        layer_state["weighted_ean_sum"],
        torch.tensor([expected_weighted_ean_sum, 0.0], dtype=torch.float64),
    )
    assert torch.allclose(
        layer_state["weighted_expert_frequency_sum"],
        torch.tensor([expected_weighted_freq, 0.0], dtype=torch.float64),
    )
    assert torch.allclose(
        layer_state["ean_mean"].mean,
        torch.tensor([9.0, 0.0], dtype=torch.float32),
    )
    assert torch.allclose(
        layer_state["reap"].mean,
        torch.tensor([expected_reap, 0.0], dtype=torch.float32),
    )
    assert torch.equal(
        layer_state["max_activations"],
        torch.tensor([12.0, 0.0], dtype=torch.float32),
    )


def test_update_pruning_state_renormalizes_selected_router_weights():
    layer_state = initialize_pruning_state(3)

    activations = torch.tensor(
        [
            [[3.0, 4.0], [8.0, 15.0]],
            [[0.0, 6.0], [0.0, 1.0]],
            [[1.0, 0.0], [7.0, 24.0]],
        ]
    )
    selected_experts = torch.tensor([[0, 1], [2, 0]], dtype=torch.long)
    router_logits = torch.tensor(
        [[2.0, 1.0, 0.0], [1.0, 0.0, 2.0]], dtype=torch.float32
    )

    pruning_batch = update_pruning_state(
        layer_state,
        activations=activations,
        selected_experts=selected_experts,
        router_logits=router_logits,
        num_experts=3,
        renormalize_router_weights=True,
    )

    routing_weights = torch.softmax(router_logits, dim=1)
    topk_weights = torch.gather(routing_weights, 1, selected_experts)
    renormalized_weights = routing_weights / topk_weights.sum(dim=-1, keepdim=True)

    expert0_weighted_ean = 5.0 * renormalized_weights[0, 0] + 17.0 * renormalized_weights[1, 0]
    expert1_weighted_ean = 6.0 * renormalized_weights[0, 1]
    expert2_weighted_ean = 25.0 * renormalized_weights[1, 2]

    assert torch.equal(pruning_batch.expert_frequency, torch.tensor([2, 1, 1]))
    assert torch.equal(
        layer_state["pairwise_expert_frequency"],
        torch.tensor([[4, 3, 3], [3, 2, 2], [3, 2, 2]]),
    )
    assert torch.allclose(
        layer_state["ean_sum"],
        torch.tensor([22.0, 6.0, 25.0], dtype=torch.float64),
    )
    assert torch.allclose(
        layer_state["ean_mean"].mean,
        torch.tensor([11.0, 6.0, 25.0], dtype=torch.float32),
    )
    assert torch.allclose(
        layer_state["weighted_expert_frequency_sum"],
        torch.tensor(
            [
                renormalized_weights[0, 0] + renormalized_weights[1, 0],
                renormalized_weights[0, 1],
                renormalized_weights[1, 2],
            ],
            dtype=torch.float64,
        ),
    )
    assert torch.allclose(
        layer_state["weighted_ean_sum"],
        torch.tensor(
            [expert0_weighted_ean, expert1_weighted_ean, expert2_weighted_ean],
            dtype=torch.float64,
        ),
    )
    assert torch.allclose(
        layer_state["reap"].mean,
        torch.tensor(
            [
                expert0_weighted_ean / 2.0,
                expert1_weighted_ean,
                expert2_weighted_ean,
            ],
        ),
    )
    assert torch.equal(
        layer_state["max_activations"],
        torch.tensor([15.0, 6.0, 24.0], dtype=torch.float32),
    )
