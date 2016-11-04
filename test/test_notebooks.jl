# Run the notebook tests in separate modules
module TestCantilever
    using NBInclude
    nbinclude("../examples/cantilever_3d.ipynb")
end

module TestStiffness
    using NBInclude
    nbinclude("../examples/stiffness_example.ipynb")
end