# block_info and the chunk-layout contract

Why `map_blocks` pins its inputs' chunk layout (`ChunksFreeze`) instead of
late-binding `block_info`/`block_id` payloads to the optimized layout.

## The problem

`map_blocks` with a `block_info`/`block_id` consumer bakes per-block payload
dictionaries at construction time, computed from the inputs' *advertised*
chunks. Simplification may later rewrite an input onto a different layout —
e.g. the native sliding-window reductions replace the advertised coarsened
chunks with the input's native chunks. The baked payloads then describe a
layout the tasks no longer have. Observed failure: `reduction(
sliding_window_view(x)).map_blocks(f, chunks=(1, 1))` with a `block_info`
consumer raised "Dimension 0 has N blocks, adjust_chunks specified with M
blocks"; shape-preserving drift would have mis-indexed blocks silently.

## Where things bind today

- **Payload contents** — built in `map_blocks` from `arg.chunks` at
  construction, stored in an `ArrayValuesDep` operand of the `Blockwise`.
  `ArrayValuesDep` is not an `Expr`, so simplify's operand substitution never
  rebuilds it: the array operands around it evolve, the payload rides along
  verbatim. `adjust_chunks` (from an explicit `chunks=` argument) freezes
  block *counts* the same way.
- **Payload emission** — already late. `_layer` and the frisky binary path
  (`Blockwise._frisky_layer`, which binds a `{coord: value}` lookup shim per
  `ArrayValuesDep`) both run on `_lowered_expr`, i.e. the settled tree.
  Staleness is entirely a contents problem, not an emission-time problem.
- **The block function itself** — user code, written against the layout the
  caller saw at construction (e.g. a writer binding
  `chunk-location[0] -> dates[i]` under a one-block-per-day rechunk). This
  binding can never be deferred; it is compiled into the function.

## Three designs

**A. Pin the layout at the consumer (`ChunksFreeze` — implemented).**
`map_blocks` wraps each array input in a `ChunksFreeze` when the function
consumes `block_id`/`block_info`. The node advertises the construction-time
chunks, is inert during simplify, and at lowering vanishes if the settled
layout matches or becomes a rechunk back to the frozen layout if a rewrite
changed it. Rewrites below the boundary still fire (the sliding-window
subtree still goes native); only the boundary re-asserts the promise, and
only when drift actually happened.

**B. Late-bind the payload contents to the settled layout.** Derive
`block_info` at layer/emission time from the current args' chunks. The
payload is then always *accurate* — but about a layout the user function
never agreed to. `block_info` is an interface between the graph and user
code, and the user-code side binds at construction; late-binding one side of
an interface doesn't change when the other side bound. For layout-sensitive
consumers (the writer above — the reason `block_info` exists) this converts
today's loud count-mismatch error into silently writing day files with the
wrong data. Late binding alone is therefore *worse* than the bug it fixes.
It is only sound when something guarantees settled == constructed — which is
exactly design A, at which point late binding is observationally identical
to baking and is pure refactor risk (Blockwise would have to rebuild
payloads in both `_layer` and `_frisky_layer`, plus late `adjust_chunks`).

**C. Gate chunk-changing rewrites on grid-sensitive dependents.** The
machinery half-exists: `Blockwise._requires_grid_preservation` is already
true for `map_blocks`-built Blockwise (`align_arrays=False`), and
slice/rechunk/shuffle *pushdowns* decline via `_preserve_grid_contract`. The
drift leaks through because algebraic rewrites (`SlidingWindowView.
_simplify_up` replacing the parent reduction) don't consult it. Extending
the gate would suppress the native sliding-window rewrite entirely under any
block_info consumer — keeping window-coarsened chunks, the exact memory
problem the native rewrite exists to fix — and puts a remember-to-check
obligation on every future chunk-changing rewrite. A is one local barrier
that makes all rewrites, present and future, safe by construction.

## The contract

`map_blocks` promises a `block_info`/`block_id` consumer **the layout it was
constructed against**. This is the only coherent choice: the function's own
layout assumptions bind at construction. `ChunksFreeze` is the mechanism that
makes the promise hold; with it in place, "construction layout" and "settled
layout" provably coincide, so the baked-vs-late-bound distinction has no
observable meaning — we keep baking because it is the code we already have.

## When late binding would become worth it

- A consumer class that is layout-*insensitive* but info-consuming (uses only
  `array-location` for absolute positioning, never block identity). For such
  functions late-bound info over a drifted layout is correct and would skip
  the bridge rechunk. If one ever matters, an explicit opt-out of the freeze
  is the right shape — not a change of default.
- Payload size: the baked dict is one nested dict per block inside an
  expression operand; for very large grids that inflates expressions and
  tokenization. Late-building the dict at emission (with the freeze still in
  place, so contents are unchanged) would be a pure optimization.
