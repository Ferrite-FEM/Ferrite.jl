"""
    extract(edof, a)

Extracts the element displacements from the global solution vector `a`
given an `edof` matrix. This assumes all elements to have the same
number of dofs.
"""
function extract(edof::VecOrMat, a::VecOrMat)

    neldofs, nel = size(edof);
    neldofs -=  1

    eldisp = zeros(neldofs, nel)

    for el = 1:nel
        eldisp[:, el] = a[edof[2:end, el]]
    end
    return eldisp
end
