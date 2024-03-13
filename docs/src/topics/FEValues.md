# [FEValues](@id fevalues_topicguide)
A key type of object in `Ferrite.jl` is the so-called `FEValues`, where the most common ones are `CellValues` and `FaceValues`. These objects are used inside the element routines and are used to query the integration weights, shape function values and gradients, and much more; see [`CellValues`](@ref) and [`FaceValues`](@ref). For these values to be correct, it is necessary to reinitialize these for the current cell by using the [`reinit!`](@ref) function. This function maps the values from the reference cell to the actual cell, a process described in detail below, see [Mapping of finite elements](@ref mapping-theory). After that, we show an implementation of a [`SimpleCellValues`](@ref SimpleCellValues) type to illustrate how `CellValues` work for the most standard case, excluding the generalizations and optimization that complicates the actual code.

## [Mapping of finite elements](@id mapping_theory)
The shape functions and gradients stored in an `FEValues` object, are reinitialized for each cell by calling the `reinit!` function. 
The main part of this calculation, considers how to map the values and derivatives of the shape functions, 
defined on the reference cell, to the actual cell.

The geometric mapping of a finite element from the reference coordinates to the real coordinates is shown in the following illustration. 

![mapping_figure](https://raw.githubusercontent.com/Ferrite-FEM/Ferrite.jl/gh-pages/assets/fe_mapping.svg)

This mapping is given by the geometric shape functions, $\hat{N}_i^g(\boldsymbol{\xi})$, such that 
```math
\begin{align*}
    \boldsymbol{x}(\boldsymbol{\xi}) =& \sum_{\alpha=1}^N \hat{\boldsymbol{x}}_\alpha \hat{N}_\alpha^g(\boldsymbol{\xi}) \\
    \boldsymbol{J} :=& \frac{\mathrm{d}\boldsymbol{x}}{\mathrm{d}\boldsymbol{\xi}} = \sum_{\alpha=1}^N \hat{\boldsymbol{x}}_\alpha \otimes \frac{\mathrm{d} \hat{N}_\alpha^g}{\mathrm{d}\boldsymbol{\xi}}\\
    \boldsymbol{\mathcal{H}} :=&
    \frac{\mathrm{d} \boldsymbol{J}}{\mathrm{d} \boldsymbol{\xi}} = \sum_{\alpha=1}^N \hat{\boldsymbol{x}}_\alpha \otimes \frac{\mathrm{d}^2 \hat{N}^g_\alpha}{\mathrm{d} \boldsymbol{\xi}^2}
\end{align*}
```
where the defined $\boldsymbol{J}$ is the jacobian of the mapping, and in some cases we will also need the corresponding hessian, $\boldsymbol{\mathcal{H}}$ (3rd order tensor).

We require that the mapping from reference coordinates to real coordinates is [diffeomorphic](https://en.wikipedia.org/wiki/Diffeomorphism), meaning that we can express $\boldsymbol{x} = \boldsymbol{x}(\boldsymbol{\xi}(\boldsymbol{x}))$, such that
```math
\begin{align*}
    \frac{\mathrm{d}\boldsymbol{x}}{\mathrm{d}\boldsymbol{x}} = \boldsymbol{I} &= \frac{\mathrm{d}\boldsymbol{x}}{\mathrm{d}\boldsymbol{\xi}} \cdot \frac{\mathrm{d}\boldsymbol{\xi}}{\mathrm{d}\boldsymbol{x}} 
    \quad\Rightarrow\quad 
    \frac{\mathrm{d}\boldsymbol{\xi}}{\mathrm{d}\boldsymbol{x}} = \left[\frac{\mathrm{d}\boldsymbol{x}}{\mathrm{d}\boldsymbol{\xi}}\right]^{-1} = \boldsymbol{J}^{-1}
\end{align*}
```
Depending on the function interpolation, we may want different types of mappings to conserve certain properties of the fields. This results in the different mapping types described below.

### Identity mapping
`Ferrite.IdentityMapping`

For scalar fields, we always use scalar base functions. For tensorial fields (non-scalar, e.g. vector-fields), the base functions can be constructed from scalar base functions, by using e.g. `VectorizedInterpolation`. From the perspective of the mapping, however, each component is mapped as an individual scalar base function. And for scalar base functions, we only require that the value of the base function is invariant to the element shape (real coordinate), and only depends on the reference coordinate, i.e. 
```math
\begin{align*}
    N(\boldsymbol{x}) &= \hat{N}(\boldsymbol{\xi}(\boldsymbol{x}))\nonumber \\
    \mathrm{grad}(N(\boldsymbol{x})) &= \frac{\mathrm{d}\hat{N}}{\mathrm{d}\boldsymbol{\xi}} \cdot \boldsymbol{J}^{-1}
\end{align*}
```

Second order gradients of the shape functions are computed as

```math
\begin{align*} 
    \mathrm{grad}(\mathrm{grad}(N(\boldsymbol{x}))) = \frac{\mathrm{d}^2 N}{\mathrm{d}\boldsymbol{x}^2} = \boldsymbol{J}^{-T} \cdot \frac{\mathrm{d}^2\hat{N}}{\mathrm{d}\boldsymbol{\xi}^2} \cdot \boldsymbol{J}^{-1} -  \boldsymbol{J}^{-T} \cdot\mathrm{grad}(N) \cdot \boldsymbol{\mathcal{H}}  \cdot \boldsymbol{J}^{-1}
\end{align*}
```
!!! details "Derivation"
    The gradient of the shape functions is obtained using the chain rule:

    \begin{align*} 
        \frac{\mathrm{d} N}{\mathrm{d}x_i} = \frac{\mathrm{d} \hat N}{\mathrm{d} \xi_r}\frac{\mathrm{d} \xi_r}{\mathrm{d} x_i} = \frac{\mathrm{d} \hat N}{\mathrm{d} \xi_r} J^{-1}_{ri}
    \end{align*}

    For the second order gradients, we first use the product rule on the equation above:

    \begin{align*} 
        \frac{\mathrm{d}^2 N}{\mathrm{d}x_i \mathrm{d}x_j} = \frac{\mathrm{d}}{\mathrm{d}x_j}(\frac{\mathrm{d} \hat N}{\mathrm{d}   \xi_r}) J^{-1}_{ri} + \frac{\mathrm{d} \hat N}{\mathrm{d} \xi_r} \frac{\mathrm{d}}{\mathrm{d}x_j}(J^{-1}_{ri}) 
    \end{align*}

    The first term can be computed as:

    \begin{align*} 
        \frac{\mathrm{d}}{\mathrm{d}x_j}(\frac{\mathrm{d} \hat N}{\mathrm{d} \xi_r}) J^{-1}_{ri} = J^{-1}_{si}\frac{\mathrm{d}}{\mathrm{d}\xi_s}(\frac{\mathrm{d} \hat N}{\mathrm{d} \xi_r}) J^{-1}_{ri} = J^{-1}_{si}\frac{\mathrm{d}^2 \hat N}{\mathrm{d} \xi_s\mathrm{d} \xi_r} J^{-1}_{ri}
    \end{align*}

    The second term can be written as:

    \begin{align*} 
        \frac{\mathrm{d} \hat N}{\mathrm{d} \xi_r}\frac{\mathrm{d}}{\mathrm{d}x_j}(J^{-1}_{ri}) = \frac{\mathrm{d} \hat N}{\mathrm{d} \xi_r}[\frac{\mathrm{d}J^{-1}_{ri}}{\mathrm{d}\xi_s}]\frac{\mathrm{d} \xi_s}{\mathrm{d}x_j} = \frac{\mathrm{d} \hat N}{\mathrm{d} \xi_r}[ - J^{-1}_{ri}\mathcal{H}_{ris} J^{-1}_{li}]J^{-1}_{sj} = - \frac{\mathrm{d} \hat N}{\mathrm{d} x_i}\mathcal{H}_{ris} J^{-1}_{li}J^{-1}_{sj} 
    \end{align*}

    where we have used that the inverse of the jacobian can be computed as:
    
    \begin{align*} 
    0 = \frac{\mathrm{d}}{\mathrm{d}\xi_s} (J_{rl} J^{-1}_{li} ) = \frac{\mathrm{d}J_{rl}}{\mathrm{d}\xi_s} J^{-1}_{li}  + J_{ri} \frac{\mathrm{d}J^{-1}_{li}}{\mathrm{d}\xi_s} = 0 \quad \Rightarrow \\
    \end{align*}

    \begin{align*} 
    \frac{\mathrm{d}J^{-1}_{li}}{\mathrm{d}\xi_s} = - J^{-1}_{ri}\frac{\mathrm{d}J_{rl}}{\mathrm{d}\xi_s} J^{-1}_{li} = - J^{-1}_{ri}\mathcal{H}_{ris} J^{-1}_{li}\\
    \end{align*}


### Covariant Piola mapping, H(curl)
`Ferrite.CovariantPiolaMapping`

The covariant Piola mapping of a vectorial base function preserves the tangential components. For the value, the mapping is defined as 
```math
\begin{align*}
    \boldsymbol{N}(\boldsymbol{x}) = \boldsymbol{J}^{-\mathrm{T}} \cdot \hat{\boldsymbol{N}}(\boldsymbol{\xi}(\boldsymbol{x}))
\end{align*}
```
which yields the gradient,
```math
\begin{align*}
    \mathrm{grad}(\boldsymbol{N}(\boldsymbol{x})) &= \boldsymbol{J}^{-T} \cdot \frac{\mathrm{d} \hat{\boldsymbol{N}}}{\mathrm{d} \boldsymbol{\xi}} \cdot \boldsymbol{J}^{-1} - \boldsymbol{J}^{-T} \cdot \left[\hat{\boldsymbol{N}}(\boldsymbol{\xi}(\boldsymbol{x}))\cdot \boldsymbol{J}^{-1} \cdot \boldsymbol{\mathcal{H}}\cdot \boldsymbol{J}^{-1}\right]
\end{align*}
```

!!! details "Derivation"
    Expressing the gradient, $\mathrm{grad}(\boldsymbol{N})$, in index notation,
    ```math
    \begin{align*}
        \frac{\mathrm{d} N_i}{\mathrm{d} x_j} &= \frac{\mathrm{d}}{\mathrm{d} x_j} \left[J^{-\mathrm{T}}_{ik} \hat{N}_k\right] = \frac{\mathrm{d} J^{-\mathrm{T}}_{ik}}{\mathrm{d} x_j} \hat{N}_k + J^{-\mathrm{T}}_{ik}  \frac{\mathrm{d} \hat{N}_k}{\mathrm{d} \xi_l} J_{lj}^{-1}
    \end{align*}
    ```

    Except for a few elements, $\boldsymbol{J}$ varies as a function of $\boldsymbol{x}$. The derivative can be calculated as 
    ```math
    \begin{align*}
        \frac{\mathrm{d} J^{-\mathrm{T}}_{ik}}{\mathrm{d} x_j} &= \frac{\mathrm{d} J^{-\mathrm{T}}_{ik}}{\mathrm{d} J_{mn}} \frac{\mathrm{d} J_{mn}}{\mathrm{d} x_j} = - J_{km}^{-1} J_{in}^{-T} \frac{\mathrm{d} J_{mn}}{\mathrm{d} x_j} \nonumber \\
        \frac{\mathrm{d} J_{mn}}{\mathrm{d} x_j} &= \mathcal{H}_{mno} J_{oj}^{-1}
    \end{align*}
    ```

### Contravariant Piola mapping, H(div)
`Ferrite.ContravariantPiolaMapping`

The covariant Piola mapping of a vectorial base function preserves the normal components. For the value, the mapping is defined as 
```math
\begin{align*}
    \boldsymbol{N}(\boldsymbol{x}) = \frac{\boldsymbol{J}}{\det(\boldsymbol{J})} \cdot \hat{\boldsymbol{N}}(\boldsymbol{\xi}(\boldsymbol{x}))
\end{align*}
```
This gives the gradient
```math
\begin{align*}
    \mathrm{grad}(\boldsymbol{N}(\boldsymbol{x})) = [\boldsymbol{\mathcal{H}}\cdot\boldsymbol{J}^{-1}] : \frac{[\boldsymbol{I} \underline{\otimes} \boldsymbol{I}] \cdot \hat{\boldsymbol{N}}}{\det(\boldsymbol{J})}
    - \left[\frac{\boldsymbol{J} \cdot \hat{\boldsymbol{N}}}{\det(\boldsymbol{J})}\right] \otimes \left[\boldsymbol{J}^{-T} : \boldsymbol{\mathcal{H}} \cdot \boldsymbol{J}^{-1}\right]
    + \boldsymbol{J} \cdot \frac{\mathrm{d} \hat{\boldsymbol{N}}}{\mathrm{d} \boldsymbol{\xi}} \cdot \frac{\boldsymbol{J}^{-1}}{\det(\boldsymbol{J})}
\end{align*}
```

!!! details "Derivation"
    Expressing the gradient, $\mathrm{grad}(\boldsymbol{N})$, in index notation,
    ```math
    \begin{align*}
        \frac{\mathrm{d} N_i}{\mathrm{d} x_j} &= \frac{\mathrm{d}}{\mathrm{d} x_j} \left[\frac{J_{ik}}{\det(\boldsymbol{J})} \hat{N}_k\right] =\nonumber\\
        &= \frac{\mathrm{d} J_{ik}}{\mathrm{d} x_j} \frac{\hat{N}_k}{\det(\boldsymbol{J})} 
        - \frac{\mathrm{d} \det(\boldsymbol{J})}{\mathrm{d} x_j} \frac{J_{ik} \hat{N}_k}{\det(\boldsymbol{J})^2}
        + \frac{J_{ik}}{\det(\boldsymbol{J})}  \frac{\mathrm{d} \hat{N}_k}{\mathrm{d} \xi_l} J_{lj}^{-1} \\
        &= \mathcal{H}_{ikl} J^{-1}_{lj} \frac{\hat{N}_k}{\det(\boldsymbol{J})} 
        - J^{-T}_{mn} \mathcal{H}_{mnl} J^{-1}_{lj} \frac{J_{ik} \hat{N}_k}{\det(\boldsymbol{J})}
        + \frac{J_{ik}}{\det(\boldsymbol{J})}  \frac{\mathrm{d} \hat{N}_k}{\mathrm{d} \xi_l} J_{lj}^{-1}
    \end{align*}
    ```

## [Walkthrough: Creating `SimpleCellValues`](@id SimpleCellValues)
In the following, we walk through how to create a `SimpleCellValues` type which 
works similar to `Ferrite.jl`'s `CellValues`, but is not performance optimized and not as general. The main purpose is to explain how the `CellValues` works for the standard case of `IdentityMapping` described above. 
Please note that several internal functions are used, and these may change without a major version increment. Please see the [Developer documentation](@ref) for their documentation. 

```@eval
# Include the example here, but modify the Literate output to suit being embedded
using Literate, Markdown
base_name = "SimpleCellValues_literate"
Literate.markdown(string(base_name, ".jl"); name = base_name, execute = true, credit = false, documenter=false)
content = read(string(base_name, ".md"), String)
rm(string(base_name, ".md"))
rm(string(base_name, ".jl"))
Markdown.parse(content)
```

## Further reading
* [defelement.com](https://defelement.com/ciarlet.html#Mapping+finite+elements)
* Kirby (2017) [Kirby2017](@cite)
