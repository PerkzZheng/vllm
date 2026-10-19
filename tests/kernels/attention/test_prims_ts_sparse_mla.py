# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native-cache adapter: holes, empty rows, two pools and graph replay.

Compare the attention result to explicit gathered attention; this catches
index/scale/LSE mistakes before expensive model-level accuracy evaluation.
"""

import pytest
import torch
from vllm.platforms import current_platform

from vllm.v1.attention.backends.mla.prims_ts_utils import (
    quantized_mla_output,
    sparse_mla,
)

_supported = False
if current_platform.is_device_capability_family(100):
    try:
        from flashinfer.attention import prims_ts

        _supported = hasattr(prims_ts, "BatchSparseMLADecodePagedTSWrapper")
    except ImportError:
        pass

pytestmark = pytest.mark.skipif(
    not _supported,
    reason="Requires SM10x and FlashInfer sparse Prims-TS MLA (#5434)",
)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize(
    "two_pools,width,extra_width",
    [
        pytest.param(False, 2048, 128, id="glm-top2048"),
        pytest.param(True, 128, 512, id="dsv4-top512"),
    ],
)
def test_sparse_storage_rows_match_attention_after_graph_replay(
    dtype, two_pools, width, extra_width
):
    torch.manual_seed(17)
    device = "cuda"
    rows, heads = 19, 16
    q = torch.randn(rows, heads, 512, device=device).to(dtype)
    # Padding between pages must be represented in the storage-row indices.
    backing = torch.randn(4, 64, 512, device=device).to(dtype)
    cache = backing[:, :32]
    extra = torch.randn(4, 32, 512, device=device).to(dtype)
    idx = torch.randint(0, 128, (rows, width), dtype=torch.int32, device=device)
    idx = idx // 32 * 64 + idx % 32
    idx[:, 3::7] = -1
    lens = torch.full((rows,), width - 9, dtype=torch.int32, device=device)
    ci = torch.randint(0, 128, (rows, extra_width), dtype=torch.int32, device=device)
    ci[:, 1::5] = -1
    cl = torch.full_like(lens, extra_width - 11)
    lens[-1] = 0
    cl[-1] = 0
    sinks = torch.linspace(-1, 1, heads, device=device) if two_pools else None
    out = torch.empty_like(q, dtype=torch.bfloat16)
    bmm1, bmm2 = 0.7 * 512**-0.5, 1.3

    def run():
        return sparse_mla(
            q,
            cache,
            idx,
            lens,
            extra_kv_cache=extra if two_pools else None,
            extra_indices=ci if two_pools else None,
            extra_lengths=cl if two_pools else None,
            bmm1_scale=bmm1,
            bmm2_scale=bmm2,
            sinks=sinks,
            out=out,
            return_lse=True,
        )

    for _ in range(2):
        run()
    graph = torch.cuda.CUDAGraph()
    capture_stream = torch.cuda.Stream()
    with torch.cuda.graph(graph, stream=capture_stream):
        result, lse = run()
    # Refresh metadata and values in place; the graph must not reuse stale lists.
    idx[:, 0] = 64
    q.copy_(torch.randn_like(q, dtype=torch.bfloat16).to(dtype))
    # Another layer/shape can use the shared scratch before this graph replays.
    capture_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(capture_stream):
        short_out, short_lse = sparse_mla(
            q[:1],
            cache,
            idx[:1],
            lens[:1],
            extra_kv_cache=extra if two_pools else None,
            extra_indices=ci[:1] if two_pools else None,
            extra_lengths=cl[:1] if two_pools else None,
            bmm1_scale=bmm1,
            bmm2_scale=bmm2,
            sinks=sinks,
            return_lse=True,
        )
    torch.cuda.current_stream().wait_stream(capture_stream)
    graph.replay()

    values, masks = [], []
    sources = [(backing, idx, lens)]
    if two_pools:
        sources.append((extra, ci, cl))
    for pool, indices, lengths in sources:
        valid = (indices >= 0) & (
            torch.arange(indices.shape[1], device=device) < lengths[:, None]
        )
        values.append(pool.reshape(-1, 512)[indices.clamp(min=0).long()].float())
        masks.append(valid)
    values = torch.cat(values, dim=1)
    valid = torch.cat(masks, dim=1)
    logits = torch.bmm(q.float(), values.transpose(1, 2)) * bmm1
    logits.masked_fill_(~valid[:, None], -torch.inf)
    expected_lse = logits.logsumexp(-1)
    normalizer = expected_lse if sinks is None else torch.logaddexp(expected_lse, sinks)
    normalizer = normalizer.masked_fill(~valid.any(-1)[:, None], 0)
    weights = (logits - normalizer[..., None]).exp()
    expected = torch.bmm(weights, values) * bmm2
    # E4M3 P has a 1/16 relative rounding bound. Bound error by the absolute
    # weighted values so cancellation near zero does not weaken the check.
    bound = torch.bmm(weights, values.abs()) * bmm2
    tolerance = (0.07 if dtype == torch.float8_e4m3fn else 0.015) * bound + 1e-3
    assert torch.all((result.float() - expected).abs() <= tolerance)
    assert torch.all((short_out.float() - expected[:1]).abs() <= tolerance[:1])
    torch.testing.assert_close(lse, expected_lse, atol=2e-3, rtol=2e-3)
    torch.testing.assert_close(short_lse, expected_lse[:1], atol=2e-3, rtol=2e-3)


@pytest.mark.parametrize(
    "dtype,block", [(torch.bfloat16, 32), (torch.float8_e4m3fn, 128)]
)
def test_fused_rope_quant_preserves_padded_output_and_replays_positions(dtype, block):
    torch.manual_seed(21)
    rows, heads, groups = 3, 16, 2
    q = (torch.randn(rows, heads, 512, device="cuda") * 0.25).to(dtype)
    kv = (torch.randn(2, 64, 512, device="cuda") + 0.5).to(dtype)
    idx = torch.randint(0, 128, (rows, 128), device="cuda", dtype=torch.int32)
    lens = torch.tensor([128, 33, 0], device="cuda", dtype=torch.int32)
    positions = torch.tensor([1, 7, 11], device="cuda", dtype=torch.int64)
    angles = (
        torch.arange(32, device="cuda")[:, None] * 0.17
        + torch.arange(32, device="cuda") * 0.03
    )
    rope = torch.cat((angles.cos(), angles.sin()), -1).contiguous()
    # The output is allocated for a larger graph bucket than the live Q prefix.
    output = quantized_mla_output(8, heads, groups, block, q.device)
    assert output.data.transpose(0, 1).is_contiguous()
    output.data.view(torch.uint8).fill_(123)
    output.scale.fill_(123)

    def run():
        return sparse_mla(
            q,
            kv,
            idx,
            lens,
            bmm1_scale=512**-0.5,
            out=output,
            output_quant_block_size=block,
            token_positions=positions,
            cos_sin_cache=rope,
        )

    for _ in range(2):
        run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result, _ = run()
    positions.add_(1)
    graph.replay()

    values = kv.reshape(-1, 512)[idx.long()].float()
    valid = torch.arange(128, device="cuda") < lens[:, None]
    scores = torch.bmm(q.float(), values.transpose(1, 2)) * 512**-0.5
    scores.masked_fill_(~valid[:, None], -torch.inf)
    weights = torch.softmax(scores, -1).nan_to_num()
    expected = torch.bmm(weights, values)
    bound = (
        torch.bmm(weights, values.abs())
        * (0.08 if dtype == torch.float8_e4m3fn else 0.016)
        + 1e-3
    )
    cos, sin = rope[positions].chunk(2, dim=-1)
    cos, sin = cos[:, None], sin[:, None]
    for tensor, error_bound in [(expected, False), (bound, True)]:
        x, y = tensor[..., 448::2].clone(), tensor[..., 449::2].clone()
        c, s = (cos.abs(), sin.abs()) if error_bound else (cos, sin)
        tensor[..., 448::2] = x * c + y * s
        tensor[..., 449::2] = y * c + x * s if error_bound else y * c - x * s
    words = output.scale[:rows]
    exponents = torch.stack([(words >> (8 * i)) & 255 for i in range(4)], -1)
    scales = torch.exp2(exponents.reshape(rows, groups, -1).float() - 127)
    actual = (
        result.float().reshape(rows, groups, -1, block) * scales[..., None]
    ).reshape(rows, heads, 512)
    assert torch.all(
        (actual - expected).abs() <= 1.07 * bound + 0.07 * expected.abs() + 1e-3
    )
    assert torch.all(output.data[rows:].view(torch.uint8) == 123)
    assert torch.all(output.scale[rows:] == 123)
