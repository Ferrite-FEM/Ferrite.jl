# Run the notebook tests in separate modules


module TestStiffness
    using NBInclude
    nbinclude("../examples/stiffness_example.ipynb")
end

module Cantilever
    using NBInclude
    nbinclude("../examples/cantilever.ipynb")
end

module HyperElasticity
    using NBInclude
    nbinclude("../examples/hyperelasticity.ipynb")
end

module HeatSquare
    using NBInclude
    # Requires BlockArrays, only available on 0.6
    # nbinclude("../examples/heat_square.ipynb")
end

module Helmholtz
    using NBInclude
    nbinclude("../examples/helmholtz.ipynb")
end

module Cook
    using NBInclude
    nbinclude("../examples/cooks_membrane_mixed_up.ipynb")
end


