"""
Extracts the element displacements from the global solution vector given an edof matrix.
This assumes all elements to have the same number of dofs.
"""
function extract_eldisp(edof::VecOrMat, a::VecOrMat)

    (nel, temp) = size(edof);
    neldofs = temp - 1

    eldisp = zeros(nel, neldofs)

    for el = 1:nel
        eldisp[el, :] = a[edof[el, 2:end]]
    end
    return eldisp
end

const extract = extract_eldisp # for CALFEM API compatability

