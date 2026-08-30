# QSA PrimTS integration

This integration keeps the QSA indexer's existing fixed-width token output and
adapts it to FlashInfer PrimTS native paged-KV metadata. It does not add a
kernel-specific top-k argument.

## Metadata route

Each query token is flattened into one independent `SQ=1` attention row. For
the model configuration (`token_topk=2048`, compression ratio and semantic page
size `4`), the indexer returns `[R, 2051]` logical token IDs:

- up to 512 complete selected groups, with four adjacent IDs per group;
- an optional causal tail of one to three IDs;
- `-1` padding after the live prefix.

`qsa_build_page4_paged_metadata` reads only column `4 * page_rank`, maps that
logical token through the request's existing vLLM block table, and writes the
encoded locator expected by FlashInfer:

```text
subpages_per_storage_page = storage_page_size / 4
locator = physical_page * subpages_per_storage_page + token_offset / 4
```

The generated tensors are:

```text
paged_kv_indptr: [R + 1]
paged_kv_indices: [R * 513]
seq_lens: [R]
```

CSR rows have fixed capacity 513, so `indptr[r] = r * 513`. `seq_lens[r]`
selects the live prefix: `min(floor((position + 1) / 4), 512) * 4` complete
tokens plus `(position + 1) % 4` tail tokens. This makes the same buffers usable
for decode, MTP, variable-length prefill, and CUDA graph replay without a host
synchronization or per-forward allocation.

The eventual PrimTS call uses `mask_type="causal"`. A flattened row's compact
query offset is `seq_lens[r] - 1`; therefore complete selected groups are fully
visible and only the one-to-three-token tail needs an element mask once the
2048-token selection budget is saturated. Earlier rows use the same causal path
and mask their shorter compact K/V extent.

CUDA-graph padding rows have logical position `-1`. The adapter gives them a
one-token placeholder row using locator zero to preserve PrimTS's positive
sequence-length contract. The attention owner must zero these inactive output
rows after the launch.

## Validation status

The metadata kernel is covered on SM103 with both 16-token and 256-token
physical cache pages. The tests include early causal positions, all tail sizes,
the 2048-token saturation boundary, encoded subpages, variable rows, and an
inactive graph-padding row.

The next checkpoint wires preallocated metadata/workspace buffers and the
PrimTS causal launch into the QSA attention owner, with the existing Triton
attention retained as a controlled fallback until end-to-end validation is
complete.
