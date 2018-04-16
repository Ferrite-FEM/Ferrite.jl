# generate examples
try
    Pkg.clone("https://github.com/fredrikekre/Examples.jl.git")
end
import Examples

EXAMPLEDIR = joinpath(@__DIR__, "src", "examples")
GENERATEDDIR = joinpath(@__DIR__, "src", "examples", "generated")
for example in readdir(EXAMPLEDIR)
    endswith(example, ".jl") || continue
    input = abspath(joinpath(EXAMPLEDIR, example))
    rmindent(str) = replace(str, r"^\h*(#'.*)$"m => s"\1")
    script = Examples.script(input, GENERATEDDIR, preprocess = rmindent)
    code = strip(read(script, String))
    mdpost(str) = replace(str, "@__CODE__" => code)
    Examples.markdown(input, GENERATEDDIR,#= preprocess = rmindent,=# postprocess = mdpost)
    Examples.notebook(input, GENERATEDDIR, execute = true)
end

# copy some figures to the build directory
cp(joinpath(@__DIR__, "../examples/figures/heat_square.png"),
   joinpath(@__DIR__, "src/examples/generated/heat_equation.png");
   remove_destination = true)
