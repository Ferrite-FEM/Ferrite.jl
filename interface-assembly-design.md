# Interface assembly redesign — Ferrite.jl (revised)

Target: [Ferrite-FEM/Ferrite.jl#1433](https://github.com/Ferrite-FEM/Ferrite.jl/issues/1433) —
assembly over interfaces when one or more fields use interpolations whose dofs are shared
between the two neighboring cells.

**Revision status.** This is the post-review revision. The previous draft (published as a
[gist](https://gist.github.com/fredrikekre/c8e251895240eaac24cef71c4b107ff8)) presented four
options (A / B / B′ / C) and recommended B′; the external review
(`INTERFACE_ASSEMBLY_DESIGN_REVIEW.md`, 2026-08-19) requested substantial changes. This
revision adopts the review's correctness blockers in full and lands on **revised Option B as
v1** — stacked kernels, condensation at an explicit boundary that leaves the assemblers
untouched — with **Option C as an opt-in v2**. A and B′ are recorded as rejected in the
appendix, and the review's decision checklist is answered at the end.

## Problem

Interface assembly for coupled problems (continuous `u`, discontinuous `P`) is broken/awkward:
1. **Sizing**: no way to query local interface matrix size. Tutorial hardcodes `zeros(2n, 2n)`.
2. **Duplicate dofs**: `interfacedofs(ic)` is a concatenation of the two cells' dofs, so shared
   dofs appear twice, and the sparse assemblers' sorted merge walk (`_assemble_inner!`,
   `src/assembler.jl:376`) mishandles the repeated index (details under Key facts — the failure
   is storage-pattern-dependent).
3. **Double counting**: `InterfaceValues` exposes a *duplicated basis*, so a shared dof has two
   local indices. Even with a duplicate-safe scatter, a kernel that gives the two copies weights
   not representing the intended operator computes the wrong local form.

These are THREE separate problems. Do not conflate 2 and 3 — fixing the scatter does not fix the
weighting. In particular (review §1.1): condensation computes `Tᵀ Ke T`; it *preserves* the
local form the kernel supplied, it cannot repair a wrongly constructed `Ke`.

## Key facts

**The `T` formulation is the specification.** Let `T` be the duplication map from unique to
stacked interface coefficients. Then:

- gather into the stacked coefficient space: `u_s = T * u_u`
- residual condensation: `f_u = Tᵀ * f_s`
- tangent condensation: `K_u = Tᵀ * K_s * T`

This vocabulary separates storage/scatter correctness (test the congruence transform on
arbitrary inputs) from weak-form correctness (test forms against hand-assembled references).

**Design fork.** You cannot have both (a) contiguous per-side blocks (`here = 1:n_a`,
`there = n_a+1:n_a+n_b`), which keeps `InterfaceValues` basis index == local matrix index, and
(b) unique local dof indices. v1 keeps (a) everywhere the user computes; (b) exists only
behind the condensation boundary. (The opt-in v2 merged view picks (b), explicitly.)

**Double counting is 4x for matrices, 2x for vectors.** 1D example: cells A=[0,1], B=[1,2],
continuous P1 `u`, shared dof `D` at x=1 appearing at stacked indices 2 and 3. Form
`a(v,u) = v(1)u(1)`, true `K[D,D] = 1`. A kernel that writes `Ke[2,2]=Ke[2,3]=Ke[3,2]=Ke[3,3]=1`
has *chosen* weight 1 for each copy; `Tᵀ Ke T` then correctly reports `K[D,D] = 4` for that
(wrong) local form. The fix is the kernel's weights, never the condensation.

**Invariant that makes the duplicated basis safe:** across the two copies of a shared dof, the
weights the kernel applies must sum to **1** for value-like operators and **0** for jump-like
ones. `{{.}}` gives 1/2+1/2=1. `[[.]]` gives -1+1=0. Raw side values summed over both copies
give 2 (and 4 squared). This is why the existing DG tutorial is correct. Raw one-sided values
are perfectly valid *when the form asks for one side* — the error is only in summing both
copies with full weight.

**Realistic bug site is mixed problems.** `∫[[P]]{{u}}` is fine. `∫(P⁻u⁻ + P⁺u⁺)` — a natural
one-sided flux pairing — silently doubles the `u` part. Needs a regression test (one
intentionally wrong kernel, one correct kernel; the wrong one asserts that condensation does
NOT repair it).

**Dof sharing follows entity placement, NOT conformity** (review §2.1, verified against
source). `conformity(ip)` describes the function-space trace; dof sharing follows where dofs
are placed during distribution. Nonconforming elements are exactly where the two differ:
`CrouzeixRaviart` and `RannacherTurek` are `L2Conformity()` (`src/interpolations.jl:1634`,
`:1700`) yet carry facet/edge dofs (`edgedof_interior_indices`, `:1647`, `:1713`) that are
shared between cells. Therefore **no conformity-based `may_share` gate**: the unique map is
built from actual global-dof equality on every `reinit!`. If profiling ever justifies a skip,
introduce a dedicated "dofs are cell-local" trait; the equality scan remains the correctness
backstop.

**Baseline on current `master` (review §4).** The issue's original two-cell MWE now *passes*:
the CSC fully-dense-column fast path accumulates duplicate rows correctly (it indexes
`offset + sortedrowdofs[ri]` without advancing a global pointer). With ≥4 line cells the same
code enters the sparse merge walk and fails with the false missing-entry error; the
high-density binary-search path has its own pointer advancement and is also affected. So the
bug is storage-pattern-dependent: identical user code can succeed on a small/dense problem and
fail after mesh growth or a coupling-mask change — which makes this *more* important, not
less. Regression MWEs must use enough cells to leave the dense shortcut and must cover every
traversal: dense CSC, binary-search CSC, ordinary merge CSC, the CSR mirrors, symmetric,
atomic, and COO.

**Current machinery (as of `origin/master`):**
- `InterfaceCache` = `(a::FacetCache, b::FacetCache, dofs::Vector{Int})`; `reinit!` fills `dofs`
  by plain concatenation (`src/iterators.jl:188`). The docstring claims "union" — wrong.
- `dof_range(ic, field)` returns a `Tuple{UnitRange,UnitRange}` and dispatches on `ic.a.cc.dh`,
  so it errors for any multi-`SubDofHandler` handler (`src/iterators.jl:214`). Undocumented,
  tested once, used by no tutorial.
- `function_value_jump` etc. hard-require `AbstractUnitRange{Int}` dof ranges
  (`src/FEValues/InterfaceValues.jl:364,396`).
- Assembler structs are immutable with four `Vector{Int}` scratch fields; `CSRAssembler` lives
  in `ext/FerriteSparseMatrixCSR.jl`. Vector assembly and `COOAssembler` are duplicate-safe.
- `apply_assemble!` → `_apply_local!` (ConstraintHandler.jl:1762) assumes unique dofs and a
  square `Ke` with `length(global_dofs) == size(Ke, 1)`.
- The sparsity machinery is already correct: `add_interface_entries!` inserts all four
  (row-cell, col-cell) blocks over the *union* of dofs. Nothing needs to change there.
- What users must write today: `test/test_dofs.jl:823` hand-rolls unique-compression with
  `unique` + `findfirst` before assembling.

---

# v1 design: stacked kernels, explicit condensation boundary

One rule organizes v1: **every object the user computes with lives in the stacked space** —
`InterfaceValues` basis, gathered coefficients (`u[interfacedofs(ic)]`), the kernel's `Ke`/`fe`,
and the field placement map. The representation change to unique dofs happens exactly once, at
a user-visible condensation call, after which the plain existing `assemble!` scatters. The
assembler structs and the generic assembly hot path are not touched.

| Use case                          | Basis/coefficient layout | Local operator layout | Scatter path                  |
|-----------------------------------|--------------------------|-----------------------|-------------------------------|
| Existing pure DG                  | stacked                  | stacked               | existing raw `assemble!`      |
| Mixed/conforming v1               | stacked                  | stacked               | condense, then scatter unique |
| AD-produced interface Jacobian v1 | stacked in/out           | stacked               | condense, then scatter unique |
| Opt-in merged v2                  | unique merged            | unique                | existing raw `assemble!`      |

## Cache state

`InterfaceCache` becomes mutable-with-const-fields (the `FacetCache` pattern), keeping `a`,
`b`, `dofs` as today and adding:

```julia
mutable struct InterfaceCache{FC <: FacetCache}
    const a::FC
    const b::FC
    const dofs::Vector{Int}               # stacked [a.dofs; b.dofs] (existing)
    const unique_dofs::Vector{Int}        # first-occurrence order: a.dofs ++ (b-only)
    const stacked_to_unique::Vector{Int}  # length(dofs); the map T as indices
    any_shared::Bool                      # derived: length(unique_dofs) != length(dofs)
    sdh_index_a::Int                      # SubDofHandler index of each side (reinit!)
    sdh_index_b::Int
end
```

- `reinit!` sets `sdh_index_a/b` from `dh.cell_to_subdofhandler` and **always** rebuilds
  `unique_dofs`/`stacked_to_unique` from actual global-dof equality: identity prefix for
  `1:n_a`, then each `b` dof tested against `a.dofs` only (each side is internally unique) —
  O(n_a·n_b) with tiny n. No conformity gate (Key facts). `any_shared` is derived.
- **First-occurrence ordering invariant**: `unique_dofs` starts with `a.dofs` verbatim, so
  `stacked_to_unique[1:n_a]` is the identity; only the there side gets holes. Deterministic,
  independent of global dof numbering, pinned by tests.
- **Lifetime rule (documented on every accessor)**: `interfacedofs(ic)`,
  `unique_interfacedofs(ic)`, and anything returned by the condensation helper are *borrowed
  cache/buffer storage*: valid only until the next `reinit!`/next condensation call, must not
  be mutated, must be copied before storing.

Accessors (`src/iterators.jl`):

| Signature                                 | Essence                                                                                                       |
|-------------------------------------------|---------------------------------------------------------------------------------------------------------------|
| `nstacked_interface_dofs(ic) -> Int`      | `length(ic.dofs)`. Basis/coefficient/local-operator size for the stacked surface (the `Ke` dim kernels fill). |
| `nunique_interface_dofs(ic) -> Int`       | `length(ic.unique_dofs)`. Scatter/condensed-system size.                                                      |
| `unique_interfacedofs(ic) -> Vector{Int}` | Duplicate-free global dofs, first-occurrence order. Borrowed view (see lifetime rule).                        |
| `is_shared(ic, i) -> Bool`                | Whether stacked index `i` refers to a dof also carried by the other cell. For checks/tests.                   |

Both sizes exist because both are useful regardless of implementation (review §3.3); neither
is called `ndofs_per_interface`, precisely because that name was ambiguous between them.

`interfacedofs(ic)` keeps name and behavior; its docstring is corrected from "union" to
"stacked concatenation — shared dofs appear once per side" with a pointer to the condensation
helper for assembly.

### `src/Dofs/DofHandler.jl`

| Signature                                | Essence                                                                                                                                                                                                                                    |
|------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `max_nstacked_interface_dofs(dh) -> Int` | Allocation bound for the **stacked** space (and therefore for both spaces): sum of the two largest `ndofs_per_cell` over sdhs. May overestimate when the two largest sdhs never neighbor — documented as a bound. Requires `isclosed(dh)`. |

## Condensation boundary

The primary surface is an explicit helper operating on a **user-owned, user-typed buffer**:

```julia
struct InterfaceAssemblyBuffer{T}
    Kc::Vector{T}   # backs the m×m condensed matrix view
    fc::Vector{T}   # backs the condensed vector view
end
InterfaceAssemblyBuffer{T}(max_n::Int) where {T}

condense_interface!(buf, ic, Ke)     -> (udofs, Kc)
condense_interface!(buf, ic, Ke, fe) -> (udofs, Kc, fc)
```

- Computes `Kc = Tᵀ Ke T` and `fc = Tᵀ fe` as `Kc[m[i], m[j]] += Ke[i, j]`,
  `fc[m[i]] += fe[i]` with `m = ic.stacked_to_unique` — `T` is never materialized.
- `udofs === unique_interfacedofs(ic)`; `Kc`/`fc` are views into `buf`. All three are borrowed
  (lifetime rule above).
- **Fast path**: when `!ic.any_shared`, returns `(interfacedofs(ic), Ke, fe)` unchanged — no
  copy, so pure DG pays nothing. (Correct because stacked == unique then.)
- **Eltype is the user's choice**, which settles the AD story: the buffer is constructed with
  whatever `T` the local operator has (`Float64`, a `Dual`, …), independent of the cache's
  coordinate eltype and of the assembler. Differentiating a stacked residual and condensing
  the resulting stacked Jacobian is the supported, boring path.

Assembly is then the plain existing call — **no assembler struct or hot-path changes, no new
`assemble!` methods**:

```julia
udofs, Kc, fc = condense_interface!(buf, ic, Ke, fe)
assemble!(assembler, udofs, Kc, fc)
```

Notes:
- **Constraints need no new overload**: after condensation, the existing
  `apply_assemble!(assembler, ch, udofs, Kc, fc)` assumptions (unique dofs, square matching
  matrix) hold by construction.
- **`SymmetricCSCAssembler`**: condensation is a congruence transform, so a symmetric stacked
  `Ke` gives a symmetric `Kc` — but a *triangle-only-filled* stacked `Ke` does not condense
  correctly (stacked `(2,3)`/`(3,2)` both fold onto the unique diagonal). Documented rule:
  supply the full stacked matrix.
- **COO** may keep accepting raw stacked entries (its finalization performs the same duplicate
  sum), but its result is tested against the explicit congruence transform.
- **Third-party assemblers**: they receive ordinary `(udofs, Kc, fc)` — nothing to implement.
- Open sub-question (not spec'd here): an `InterfaceDofs(ic, buf) <: AbstractVector{Int}`
  wrapper (suggested in the #1433 discussion and review §5.2) that gives `assemble!` a dispatch
  point folding the two calls into one. Pure sugar over the helper; decide during
  implementation review.

## Field placement map

```julia
struct InterfaceDofRange <: AbstractVector{Int}
    here::UnitRange{Int}    # dof_range(sdh_a, field)
    there::UnitRange{Int}   # dof_range(sdh_b, field) .+ ndofs_per_cell(sdh_a)
end
Base.size(r::InterfaceDofRange) = (length(r.here) + length(r.there),)
Base.@propagate_inbounds function Base.getindex(r::InterfaceDofRange, i::Int)
    return i <= length(r.here) ? r.here[i] : r.there[i - length(r.here)]
end
```

`dof_range(ic::InterfaceCache, field::Symbol) -> InterfaceDofRange`, built per `reinit!` from
`sdh_index_a/b` (multi-`SubDofHandler` correct, unlike the current method).

- Index `i` runs over the field's duplicated basis (matches `getnbasefunctions(iv)` for that
  field's `InterfaceValues`); the value is the row/col in the **stacked** `Ke`. Under v1 this
  object is genuinely range-like — two contiguous ranges concatenated, no repeated values, no
  backward jumps. (The review's §3.2 objection — "a range that is secretly a repeated scatter
  map" — applied to the rejected B′ layout, not to this one.)
- The `function_*` evaluation methods gain unpacking overloads
  `function_value(iv, qp, u, r::InterfaceDofRange; here) = function_value(iv, qp, u, r.here,
  r.there; here)` etc. for all six methods. `InterfaceValues` stays dof-agnostic.
- **Breaking change, taken deliberately**: the existing method returns
  `Tuple{UnitRange,UnitRange}`, and `r_here, r_there = dof_range(ic, f)` would *silently*
  destructure the first two integers of the new return type. Mitigation: the CHANGELOG
  breaking entry shows `(r.here, r.there)` as the one-line port and names the destructuring
  hazard explicitly. Rationale for reuse over a new name: the old method is undocumented,
  tested once, unusable for its purpose, and `dof_range` is the name kernel authors know from
  cell assembly.
- **A field absent on one side is rejected in v1** (review §2.4): `dof_range(ic, f)` throws
  with a clear message when `f` is missing from either side's sdh. Empty ranges are *not*
  support: `InterfaceValues` interprets basis indices as here-then-there, a there-only field's
  indices would be misread as here-side, and no one-sided values representation exists.
  One-sided interfaces are future work with a real design, not an emergent property.

## `InterfaceValues`: semantics and documentation (no new evaluation API)

No trace aliases are added (review §2.3: an H1 field's *gradient* is not single-valued across
a facet — only its tangential part is; full-vector values of Hdiv/Hcurl fields are not
single-valued — only the normal/tangential traces are). The existing precise names carry the
semantics, and the documentation teaches the weighting rule in operator terms:

- `shape_*_average` / `shape_*_jump` express the correct cross-copy weights (1 and 0);
- for a field *continuous across the interface*, `shape_value_average` of the two duplicated
  copies is how the single-valued value trace is written in the duplicated basis;
- raw one-sided values are valid when the form asks for one side;
- summing two raw side contributions is neither an average nor a trace;
- condensation preserves the local form; it cannot infer or repair its weights.

(The previous draft's docstring claim "raw side values are valid only for discontinuous
fields" was wrong and is gone.) Conformity-specific trace helpers — normal trace for Hdiv,
tangential for Hcurl — are deferred; they require an explicit normal/orientation convention,
which is named as the blocker.

## Sparsity

The documented default is the full mask:

```julia
K = allocate_matrix(dh; topology, interface_coupling = trues(nfields, nfields))
```

A smaller mask is justified **only** when the implemented form has no such block *by
construction* (e.g. the mixed example below has no `u`–`u` interface term, so its `u`–`u`
interface block may be omitted). It is never derived from conformity or from numerical
cancellation: conforming fields can have genuinely nonzero cross-cell blocks (gradient
averages, cohesive/interface physics, trace reactions), and sparse *structure* must not depend
on independently evaluated floating-point terms cancelling to exact `0.0`. The previous
draft's cancellation-based "lean mask" recommendation is withdrawn.

## Diagnostics

On the missing-entry error path in `_assemble_inner!` (error path only, zero happy-path cost):
report **both** facts when both hold — the requested global entry is missing from the pattern,
*and* the index vector contains duplicated dofs (pointing at `condense_interface!`). Never
replace the missing-entry report unconditionally: duplicates do not prove the missing entry is
a false positive; a kernel may have duplicates *and* attempt a genuinely absent coupling
(review §3.6). Documented limitation: this is a crash diagnostic, not a semantic guard — on
the dense CSC path a raw duplicated scatter silently computes `Tᵀ Ke T` and no message fires.

## User surface

### (a) DG heat tutorial — kernel body byte-identical to today

```julia
K = allocate_matrix(dh; interface_coupling = [true;;], topology = topology)

Ki  = zeros(max_nstacked_interface_dofs(dh), max_nstacked_interface_dofs(dh))
buf = InterfaceAssemblyBuffer{Float64}(max_nstacked_interface_dofs(dh))
assembler = start_assemble(K, f)

for ic in InterfaceIterator(dh, topology)
    reinit!(interfacevalues, ic)
    hₑ = getdiameter(∩(getcoordinates(ic)...))
    μ = (1 + order)^dim / hₑ
    n = nstacked_interface_dofs(ic)
    Kie = @view Ki[1:n, 1:n]
    fill!(Kie, 0)
    assemble_interface!(Kie, interfacevalues, μ)     # unchanged kernel
    udofs, Kc = condense_interface!(buf, ic, Kie)    # pass-through (no copy) for pure DG
    assemble!(assembler, udofs, Kc)
end
```

For pure DG the condense call is a no-op pass-through, and the raw
`assemble!(assembler, interfacedofs(ic), Kie)` also remains valid — the uniform pattern above
is what tutorials show so that adding a conforming field later changes nothing.

### (b) Mixed continuous `u` / discontinuous `P` — the issue #1433 case

```julia
function assemble_interface!(Ke, ic, iv_u, iv_P, μ)
    rP = dof_range(ic, :P)   # InterfaceDofRange: field basis index -> stacked Ke index
    ru = dof_range(ic, :u)
    for qp in 1:getnquadpoints(iv_P)
        dΓ = getdetJdV(iv_P, qp)
        n = getnormal(iv_P, qp)
        for (i, I) in pairs(rP)
            δP_jump = shape_value_jump(iv_P, qp, i)
            # ∫ μ [[δP]][[P]] dΓ — cell-local field: jumps as usual
            for (j, J) in pairs(rP)
                Ke[I, J] += μ * δP_jump * shape_value_jump(iv_P, qp, j) * dΓ
            end
            # ∫ [[δP]] (u ⋅ n) dΓ + transpose. u is continuous across the interface, so
            # the two duplicated copies of a shared dof must receive weights summing to 1:
            # shape_value_average gives each copy 1/2, and after condensation the pair
            # contributes the single-valued trace of u. Summing both raw side values
            # (weight 1 each) would double-count shared dofs — and condensation would
            # faithfully assemble that wrong form.
            for (j, J) in pairs(ru)
                uj = shape_value_average(iv_u, qp, j) ⋅ n
                Ke[I, J] += δP_jump * uj * dΓ
                Ke[J, I] += δP_jump * uj * dΓ
            end
        end
    end
    return Ke
end

Ke  = zeros(max_nstacked_interface_dofs(dh), max_nstacked_interface_dofs(dh))
Fi  = zeros(max_nstacked_interface_dofs(dh))
buf = InterfaceAssemblyBuffer{Float64}(max_nstacked_interface_dofs(dh))
assembler = start_assemble(K, f)

for ic in InterfaceIterator(dh, topo)
    reinit!(iv_u, ic); reinit!(iv_P, ic)
    n = nstacked_interface_dofs(ic)
    ke = @view Ke[1:n, 1:n]
    fie = @view Fi[1:n]
    fill!(ke, 0); fill!(fie, 0)
    assemble_interface!(ke, ic, iv_u, iv_P, μ)
    udofs, Kc, fc = condense_interface!(buf, ic, ke, fie)
    assemble!(assembler, udofs, Kc, fc)
end
```

Constrained assembly, via the *existing* method (unique dofs and matching square matrix hold
after condensation):

```julia
udofs, Kc, fc = condense_interface!(buf, ic, ke, fie)
apply_assemble!(assembler, ch, udofs, Kc, fc)
```

Post-processing with a solution vector (everything stacked):

```julia
ue = a[interfacedofs(ic)]                                       # stacked gather, duplicate-safe
jump_P = function_value_jump(iv_P, qp, ue, dof_range(ic, :P))
```

AD sketch (stacked all the way to the boundary):

```julia
ue = a[interfacedofs(ic)]
Ke = ForwardDiff.jacobian(u -> stacked_residual(u, ic, iv_u, iv_P), ue)
buf_dual = InterfaceAssemblyBuffer{eltype(Ke)}(max_nstacked_interface_dofs(dh))
udofs, Kc = condense_interface!(buf_dual, ic, Ke)
```

### (c) Sparsity pattern for the mixed case

```julia
# Field order (:u, :P). Default — all interface blocks:
K = allocate_matrix(dh; topology, interface_coupling = trues(2, 2))
```

This particular form has no `u`–`u` interface term, so `Bool[0 1; 1 1]` is *also* structurally
valid — because of the form, not because `u` is conforming. Documentation states the rule that
way and defaults to the full mask.

## What stays the same

- `InterfaceValues`: dof-agnostic, duplicated basis, `getnbasefunctions = here + there`, all
  `shape_*`/`function_*` semantics, `[[v]] = v_there − v_here`. No new evaluation API in v1.
- `interfacedofs(ic)`, `InterfaceIterator`, `reinit!(iv, ic)` — same names, same behavior
  (only the `interfacedofs` docstring is corrected).
- Every existing DG kernel and assembly loop works verbatim; adopting the condense boundary is
  optional for pure DG (pass-through, zero overhead).
- Sparsity-pattern machinery — unchanged.
- Assembler structs, `start_assemble`, the generic `assemble!` hot loop, both extensions —
  unchanged except the error-path diagnostic.

## Exports (new)

`nstacked_interface_dofs`, `nunique_interface_dofs`, `max_nstacked_interface_dofs`,
`unique_interfacedofs`, `is_shared`, `InterfaceAssemblyBuffer`, `condense_interface!`.
(`InterfaceDofRange` is the documented return type of `dof_range(ic, field)`, constructor not
exported.)

---

# v2: merged interface basis (`MergedInterfaceValues`, opt-in)

deal.II-style: one local index per unique field dof, so duplicated-basis double counting is
structurally unavailable — the only surface that can make the §(b) weighting comment
unnecessary. It stays an **opt-in wrapper** over the same cache state; the stacked
`InterfaceValues` surface is not deprecated by it, and the DG tutorial stays stacked. A new
mixed-interface how-to teaches the merged surface once it exists.

Sketch (an **allocation-free wrapper** — not "zero cost": every there-side evaluation pays a
table lookup plus a branch, which blocks SIMD across `i` and gets a benchmark):

```julia
struct MergedInterfaceValues{IV <: InterfaceValues, IC <: InterfaceCache} <: AbstractValues
    iv::IV
    ic::IC
    # field resolution + maps below
end
MergedInterfaceValues(iv, ic, field::Symbol)
```

Indexing needs more than the previous draft's single `merged_to_b` table (review §3.5): three
coordinate systems are in play (whole-interface stacked positions; each side's field-local
basis positions; field-local merged positions), and "the a-side representative is simply `i`"
holds only after field offsets and presence are resolved. The specification therefore uses two
explicit per-wrapper maps plus a placement map:

```text
merged basis i -> a-side field-local basis index, or 0 if absent
merged basis i -> b-side field-local basis index, or 0 if absent
field-local merged i -> whole-interface unique matrix position
```

Evaluation combines the (up to two) representatives; for a shared dof, `average` is the value
trace by construction and any contribution lands in one row/column. **No exact-zero jump fast
path**: it would be valid only for H1 value traces, and Hdiv/Hcurl shared dofs have genuinely
nonzero full-vector jumps (only their normal/tangential traces are continuous — the Darcy
tutorial exercises this).

Open items to resolve before this becomes public API:

- field-local vs whole-interface offsets, nailed down for multiple fields;
- fields absent on one side (needs a real one-sided values model, cf. v1's rejection);
- different interpolation orders/types for the same field across SubDofHandlers;
- H1 value traces vs gradient averages (the merged view must not imply gradients are traces);
- normal/tangential Hdiv/Hcurl traces and their orientation conventions;
- hanging-node constraints and nonconforming interfaces;
- cache-view lifetime (a debug generation counter on the wrapper is worth considering);
- complete `shape_*`/`function_*` coverage and benchmarked hot-loop cost.

Assembly for the merged view is the plain existing `assemble!` over
`unique_interfacedofs(ic)` — no condensation, no assembler changes; that part is shared with
v1's boundary and already tested by it.

---

# Test and benchmark matrix

## Representation and storage

1. `unique_dofs`/`stacked_to_unique` verified against `unique(stacked)` for H1, Hdiv, Hcurl,
   **`CrouzeixRaviart`, `RannacherTurek`**, and true DG interpolations.
2. `Kc == Tᵀ Ke T`, `fc == Tᵀ fe` for arbitrary nonsymmetric `Ke`/`fe` (not weak-form values).
3. All storage traversals exercised with duplicated raw scatters and with the condensed path:
   fully dense CSC, high-density binary-search CSC, ordinary merge CSC, CSR mirrors,
   symmetric, atomic; ≥4-cell #1433 regression (2-cell case kept, but paired — it only hits
   the dense shortcut).
4. COO / CSC / CSR / symmetric / atomic agreement where contracts overlap; COO raw-stacked
   result equals the explicit congruence transform.
5. First-occurrence ordering and cache reuse across interfaces with different sizes and sdhs.

## Weak-form semantics

1. Mixed H1/L2 average–jump form against a hand-assembled reference.
2. The one-sided flux pairing: one intentionally wrong stacked kernel (assert condensation
   does **not** repair it) and one correct kernel.
3. An H1 gradient-average term demonstrating `shape_gradient_average` is not a single-valued
   gradient trace.
4. Raviart–Thomas normal continuity and Nedelec tangential continuity, with their full-vector
   jumps not vanishing.
5. A conforming-field interface form with a genuinely nonzero cross-cell `u`–`u` block,
   proving mask choice cannot be inferred from conformity.

## Constraints and heterogeneous domains

1. Prescribed and affine constraints on a shared interface dof; a nonlocal affine constraint
   through `apply_assemble!`.
2. Hanging-node/nonconforming interfaces.
3. Multiple SubDofHandlers with the same field and different local field orderings.
4. One-sided field: targeted rejection test (error message asserted).

## AD and performance

1. Stacked-residual Jacobian condensed vs differentiation w.r.t. unique coefficients through
   the gather `u_s = T u_u` — equal up to roundoff.
2. Benchmark: unchanged cell assembly (generic hot path untouched), pure DG interface
   assembly (pass-through overhead == 0), mixed H1/L2 assembly, and (v2) the merged wrapper.
3. Allocations per interface reported for each supported path.

---

# Appendix

## Rejected: Option A — duplicate-tolerant assembler as the fix

Make the merge walk duplicate-aware so raw stacked scatters accumulate. Rejected as the
standalone fix: it repairs storage traversal only — not sizing, not constraints
(`_apply_local!` still assumes unique dofs), and not the weighting semantics; the one-sided
flux kernel above still assembles silently wrong. It also costs a comparison per matched entry
in the hottest loop and needs delicate `SymmetricCSCAssembler` diagonal handling. Its
diagnostics idea survives in v1 (both-facts error message).

## Rejected: Option B′ — unique-mapped writes inside the kernel

`dof_range(ic, f)` returning unique-mapped indices so the kernel's own `+=` performs the fold,
with plain `assemble!` after. This was the previous draft's recommendation (motivated by
keeping condensation out of the assemblers — a goal v1 now meets with the explicit helper
instead). Rejected per review §3.1: it breaks the contract that basis/coefficient index `i` is
row/column `i` of the local operator, forces every mixed kernel to coordinate two index
spaces, turns the "range" into a repeated scatter map, leaves raw `Ke[i, j]` valid for DG but
silently invalid for sharing fields, and makes AD and local-operator composition harder — all
while the double-counting footgun remains representable. The review also identified internal
inconsistencies in its write-up (fast path skipping the map its `getindex` reads, §2.5;
diagnostics naming a method it deletes, §2.6), which are moot with the rejection.

## Review decision checklist (review §7), answered

1. v1 local operator: **stacked**.
2. `Tᵀ Ke T` location: **explicit `condense_interface!` boundary** with a user-owned typed
   buffer; assembler structs and generic hot path untouched. (`InterfaceDofs` dispatch sugar
   left as an implementation-review question.)
3. Duplicate detection from actual global ids: **yes**, on every `reinit!`; no conformity gate.
4. Both sizes named explicitly: **yes** — `nstacked_interface_dofs` / `nunique_interface_dofs`;
   the allocation bound is named for the stacked space it bounds.
5. `dof_range(ic, field)` semantics: **changed, deliberately** (deviation from the review's
   recommendation, decided by the maintainer): the tuple method is undocumented/near-unused/
   broken for multi-sdh, and under the stacked v1 the new return is genuinely range-like.
   CHANGELOG breaking entry names the silent-destructuring hazard and the `(r.here, r.there)`
   port.
6. Missing-side fields: **rejected in v1** with an explicit error; support deferred to a real
   one-sided design.
7. Sparsity relying on exact cancellation: **no** — full mask default; smaller masks justified
   only by the form's block structure.
8. Trace APIs: **none exported** in v1; future trace helpers must be conformity-specific with
   explicit orientation conventions.
9. Lifetime of cache-backed maps/views: **documented borrowed-view rule** (valid until next
   `reinit!`/condense; no mutation; copy to store); generation-counter idea noted for v2.
10. Error messages name shipping APIs: **yes** — diagnostics point to `condense_interface!`,
    which exists in v1.

## References

- Review: `INTERFACE_ASSEMBLY_DESIGN_REVIEW.md` (2026-08-19).
- Previous draft: the four-option gist
  (https://gist.github.com/fredrikekre/c8e251895240eaac24cef71c4b107ff8).
- Issue: https://github.com/Ferrite-FEM/Ferrite.jl/issues/1433 (incl. the `InterfaceDofs`
  wrapper suggestion).
- Prior art: deal.II `FEInterfaceValues`
  (https://dealii.org/9.4.0/doxygen/deal.II/classFEInterfaceValues.html) — per-interface dof
  count, duplicate-free joint index vector; its component-recovery rough edge motivates v2's
  per-field merged index space.
