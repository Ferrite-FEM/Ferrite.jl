# Run the notebook tests in separate modules

ENV["JUAFEM_TESTING"] = true

module TestStiffness
    using NBInclude
    mktempdir() do dir
        cd(dir) do
            @nbinclude("../examples/stiffness_example.ipynb")
        end
    end
end

module Cantilever
    using NBInclude
    mktempdir() do dir
        cd(dir) do
            @nbinclude("../examples/cantilever.ipynb")
        end
    end
end

module HyperElasticity
    using NBInclude
    mktempdir() do dir
        cd(dir) do
            @nbinclude("../examples/hyperelasticity.ipynb")
        end
    end
end

module HeatSquare
    using NBInclude
    mktempdir() do dir
        cd(dir) do
            @nbinclude("../examples/heat_square.ipynb")
        end
    end
end

module Helmholtz
    using NBInclude
    mktempdir() do dir
        cd(dir) do
            @nbinclude("../examples/helmholtz.ipynb")
        end
    end
end

module Cook
    using NBInclude
    mktempdir() do dir
        cd(dir) do
            @nbinclude("../examples/cooks_membrane_mixed_up.ipynb")
        end
    end
end


