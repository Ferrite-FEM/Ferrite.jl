# IDEA: Have two functions, one for current behavior and one for always evaluating to a scalar dof.


#=
| L2Projector input | ScalarInterpolation   | VectorizedInterpolation   | VectorInterpolation |
| Data type input   | -                     | -                         | - |
| `Number`          | `Number`              | `Number`                  | N/A |
| `Vec`             | `Vec`                 | `Vec`                     | `Number` |
| `Tensor{2}`       | `Tensor{2}`           | `Tensor{2}`               | N/A |
=#
