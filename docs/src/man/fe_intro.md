# Introduction to FEM
blablabla weak format

## Strong format
The strong format of equilibrium of momentum can be written as:

$-\mathbf{\sigma}\cdot \mathbf{\nabla} = \mathbf{f}$

where $\mathbf{\sigma}$ is the stress tensor and $\mathbf{f}$ is the internal force.

## Weak format

$u \in \mathbb{U}: \quad a(u,\delta u) = l(\delta u) \quad \forall \delta u \in \mathbb{U}^0$

## FE-approximation

Introduce the finite element approximation $\mathbf{u}_h$ (e.g. $\mathbf{u} \approx \mathbf{u}_\text{h}$)
as a sum of shape functions, $\mathbf{N}_i$ and nodal values, $a_i$

$\mathbf{u}_\text{h} = \sum_{i=1}^{\text{N}} \mathbf{N}_i a_i,\qquad \delta\mathbf{u}_\text{h} = \sum_{i=1}^{\text{N}} \mathbf{N}_i \delta a_i$

This approximation can be inserted in the weak form, which gives

