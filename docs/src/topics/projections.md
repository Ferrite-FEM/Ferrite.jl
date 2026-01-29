# [Projections and Transfer](@id man-projection-transfer)

This section is concerned with projection and transfer techniques implemented in Ferrite.

## [L2 Projection](@id man-l2projection)

A commonly used projection technique in finite element problems is the L2 projection which is implemented in Ferrite as the `[L2Projector](@ref)`.
This technique is concerned with finding the best field for some function, or after numerical quadrature for some given data points.
To be mathematically more precise, the operation done by the L2-projector can be interpreted as the Galerkin-projection of ``f`` onto ``u``:
```math
    \int v (u - f) \ \mathrm{d}\Omega = 0 \quad \forall v \in U(\Omega),
```
where ``U`` is the test space on ``\Omega``.
An alternative interpretation here is that we minimize the difference between ``f`` and ``u`` with the functional (induced by the ``L_2`` norm):
```math
    \min_u \int \frac{1}{2} (u - f)^2 \ \mathrm{d}\Omega \, .
```
Please note that the former projection form is equivalent to the first variation (Frechet derivative) of the minimization problem.
The final linear system can be then stated as
```math
\int v u \ \mathrm{d}\Omega = \int v f \ \mathrm{d}\Omega \quad \forall v \in U(\Omega),
```
where we can see that applying a numerical quadrature rule to the right hand side shows us that we just need some samples of ``f`` at the quadrature points for this technique to work.

After finite element discretization the resulting problem to solve becomes
```math
\underbrace{\int N_i N_j \ \mathrm{d}\Omega}_{M_{ij}}\ u^h_j = \underbrace{\int N_i f \ \mathrm{d}\Omega}_{b_i}
```
where ``u^h_j`` denotes the unknown dofs of the finite-dimensional discrete space to project onto (e.g. Lagrange polynomials) with ansatz and test functions ``N``.
``b_i`` denotes the right hand side of the finite-dimensional discrete linear system. For vectorized problems we simply apply the projection component-wise.
