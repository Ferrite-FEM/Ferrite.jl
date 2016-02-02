"""
    extract(edof, a)

Extracts the element displacements from the global solution vector `a`
given an `edof` matrix. This assumes all elements to have the same
number of dofs.
"""
function extract(edof::VecOrMat, a::VecOrMat)
    neldofs, nel = size(edof);

    eldisp = zeros(neldofs, nel)

    for el = 1:nel
        eldisp[:, el] = a[edof[:, el]]
    end
    return eldisp
end
