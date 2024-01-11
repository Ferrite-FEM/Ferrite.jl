struct QuadPointValues{VT<:AbstractValues}
    fe_v::VT
    q_point::Int
end

#= Proposed syntax in general 
for qp in QuadraturePointValuesIterator(cv::AbstractValues)
    dV = getdetJdV(qp)
    u = function_value(qp, ae)
    for i in 1:getnbasefunctions(qp)
        Ni = shape_value(qp, i)
        fe[i] += Ni*dV
        for j in 1:getnbasefunctions(qp)
            Nj = ...
            Ke[i,j] += ...
        end
    end 
end

Where the default for a QuadraturePointValuesIterator would be to return a 
`QuadPointValues` as above, but custom `AbstractValues` can be created where 
for example the element type would be a static QuadPointValue type which doesn't 
use heap allocated buffers, e.g. by only saving the cell and coordinates during reinit, 
and then calculating all values for each element in the iterator. 

References: 
https://github.com/termi-official/Thunderbolt.jl/pull/53/files#diff-2b486be5a947c02ef2a38ff3f82af3141193af0b6f01ed9d5129b914ed1d84f6
https://github.com/Ferrite-FEM/Ferrite.jl/compare/master...kam/StaticValues2
=#