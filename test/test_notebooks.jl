# Run the notebook tests in separate modules

module TestStiffness
    using NBInclude
    nbinclude("../examples/stiffness_example.ipynb")
    nbinclude("../examples/cantilever.ipynb")
end