## Strong form
$$
\nabla \cdot \boldsymbol{q} = h \in \Omega \\
\boldsymbol{q} = - k\ \nabla u \in \Omega \\
\boldsymbol{q}\cdot \boldsymbol{n} = q_n \in \Gamma_\mathrm{N}\\
u = u_\mathrm{D} \in \Gamma_\mathrm{D}
$$

## Weak form
### Part 1
$$
\int_{\Omega} \delta u \nabla \cdot \boldsymbol{q}\ \mathrm{d}\Omega = \int_{\Omega} \delta u\ h\ \mathrm{d}\Omega \\
\int_{\Gamma} \delta u \boldsymbol{n} \cdot \boldsymbol{q}\ \mathrm{d}\Gamma -
\int_{\Omega} \nabla (\delta u) \cdot \boldsymbol{q}\ \mathrm{d}\Omega = \int_{\Omega} \delta u\ h\ \mathrm{d}\Omega \\
$$

### Part 2
$$
\int_{\Omega} \boldsymbol{\delta q} \cdot \boldsymbol{q}\ \mathrm{d}\Omega = - \int_{\Omega} \boldsymbol{\delta q} \cdot \left[k\ \nabla u\right]\ \mathrm{d}\Omega
$$
where no Green-Gauss theorem is applied. 

### Summary
The weak form becomes, find $u\in H^1$ and $\boldsymbol{q} \in H\mathrm{(div)}$, such that
$$
\begin{align*}
-\int_{\Omega} \nabla (\delta u) \cdot \boldsymbol{q}\ \mathrm{d}\Omega &= \int_{\Omega} \delta u\ h\ \mathrm{d}\Omega -
\int_{\Gamma} \delta u\ q_\mathrm{n}\ \mathrm{d}\Gamma
\quad
\forall\ \delta u \in \delta H^1 \\
\int_{\Omega} \boldsymbol{\delta q} \cdot \boldsymbol{q}\ \mathrm{d}\Omega &= - \int_{\Omega} \boldsymbol{\delta q} \cdot \left[k\ \nabla u\right]\ \mathrm{d}\Omega
 \quad \forall\ \boldsymbol{\delta q} \in \delta H\mathrm{(div)}
\end{align*}
$$
