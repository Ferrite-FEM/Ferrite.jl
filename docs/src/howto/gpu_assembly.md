# [GPU Assembly](@id gpu_assembly_howto)

For some large problems it can be beneficial to use a GPU to assemble the residual and in some cases even the system.
Before developing hand-written optimiezd assembly routines using Ferrite users can try to port their existing assembly
routine.
In the following we show how users can assemble the heat problem from the first tutorial using CUDA either directly
or via KernelAbstractions. This can also serve as a first step towards more elaborate assembly optimizations on the
GPU.

```@eval
# Include the example here, but modify the Literate output to suit being embedded
using Literate, Markdown
base_name = "gpu_heat_howto_literate"
Literate.markdown(string(base_name, ".jl"); name = base_name, execute = false, credit = false, documenter=false)
content = read(string(base_name, ".md"), String)
rm(string(base_name, ".md"))
rm(string(base_name, ".jl"))
Markdown.parse(content)
```
