# Introduction to FEM
Here, a very brief introduction to the finite element method is given, for reference. As an
illustrative example we use balance of momentum, with a linear elastic material.

## Strong format

The strong format of balance of momentum can be written as:

$-\mathbf{\sigma}\cdot \mathbf{\nabla} = \mathbf{f}$

where $\mathbf{\sigma}$ is the stress tensor and $\mathbf{f}$ is the internal force (body force).
To complete the system of equations we need boundary conditions. These are normally given as known
displacements $\mathbf{u}^\text{p}$ or known traction $\mathbf{t}^\text{p}$ on different parts of the boundary.
The stress can, using the linear elastic material model, be written as

$\mathbf{\sigma} = \mathbf{\mathsf{C}} : \mathbf{\varepsilon}(\mathbf{u}) = \mathbf{\mathsf{C}} : [\mathbf{u} \otimes \mathbf{\nabla}]^{\text{sym}}$

where $\mathbf{\mathsf{C}}$ is the 4th order elasticity tensor and $\mathbf{\varepsilon}(\mathbf{u})$ the symmetric
part of the gradient of the displacement field $\mathbf{u}$.


## Weak format

The solution of the equation above is usually calculated from the corresponding weak format. By multiplying
the equation with an arbitrary test function $\mathbf{\delta u}$, integrating over the domain and using partial
integration the following equation is obtained; Find $\mathbf{u} \in \mathbb{U}$ s.t.

$\int_\Omega \mathbf{\varepsilon}(\mathbf{\delta u}) : \mathbf{\mathsf{C}} : \mathbf{\varepsilon}(\mathbf{u}) \text{d}\Omega = 
\int_\Gamma \mathbf{\delta u} \cdot \mathbf{t}^\text{p} \text{d}\Gamma +
\int_\Omega \mathbf{\delta u} \cdot \mathbf{f} \text{d}\Omega \qquad \forall \mathbf{\delta u} \in \mathbb{U}^0$

where $\mathbb{U}, \mathbb{U}^0$ are function spaces with sufficiently regular functions. The solution to this equation is identical to the one of the strong format.

## FE-approximation

Introduce the finite element approximation $\mathbf{u}_h$ (e.g. $\mathbf{u} \approx \mathbf{u}_\text{h}$)
as a sum of shape functions, $\mathbf{N}_i$ and nodal values, $a_i$

$\mathbf{u}_\text{h} = \sum_{i=1}^{\text{N}} \mathbf{N}_i a_i,\qquad \delta\mathbf{u}_\text{h} = \sum_{i=1}^{\text{N}} \mathbf{N}_i \delta a_i$

This approximation can be inserted in the weak format, which gives

$\sum_i^N\sum_j^N \delta a_i \int_\Omega \mathbf{\varepsilon}(\mathbf{N}_i) : \mathbf{\mathsf{C}} : \mathbf{\varepsilon}(\mathbf{N}_j) \text{d}\Omega a_j = 
\sum_i^N \delta a_i \int_\Gamma \mathbf{N}_i \cdot \mathbf{t}^\text{p} \text{d}\Gamma +
\sum_i^N \delta a_i \int_\Omega \mathbf{N}_i \cdot \mathbf{f} \text{d}\Omega$

Since $\mathbf{\delta u}$ is arbitrary, the nodal values $\delta a_i$ are arbitrary. Thus, the equation can
be written as

$\underline{K}\ \underline{a} = \underline{f}$

where $\underline{K}$ is the stiffness matrix, $\underline{a}$ is the solution vector with the nodal values
and $\underline{f}$ is the force vector. The elements of $\underline{K}$ and $\underline{f}$ are given by

$\underline{K}_{ij} = \int_\Omega \mathbf{\varepsilon}(\mathbf{N}_i) : \mathbf{\mathsf{C}} : \mathbf{\varepsilon}(\mathbf{N}_j) \text{d}\Omega$

$\underline{f}_{i} = \int_\Gamma \mathbf{N}_i \cdot \mathbf{t}^\text{p} \text{d}\Gamma +
                     \int_\Omega \mathbf{N}_i \cdot \mathbf{f} \text{d}\Omega$

The solution to the system (which in this case is linear) is simply given by inverting the matrix $\underline{K}$ and
using the boundary conditions (prescribed displacements)

$\underline{a} = \underline{K}^{-1}\ \underline{f}$

## Implementation in JuAFEM

In practice, the shape functions $\mathbf{N}$ are only non-zero on parts of the domain $\Omega$. Thus, the integrals
are evaluated on sub-domains, called elements or cells. All the cells gives a contribution to the global
stiffness matrix and force vector.

The integrals are evaluated using quadrature. In JuAFEM the stiffness matrix and force vector can be calculated like this

```julia
...

for qp in 1:Nqp
    for i in 1:N
        f[i] += shape_value(i) ⋅ f * dΩ
        for j in 1:N
            K[i,j] += shape_symmetric_gradient(i) : C : shape_symmetric_gradient(j) * dΩ
        end
    end
end

...
```

Although this is a simplification of the actual code, note the similarity between the code and the mathematical expression above.
