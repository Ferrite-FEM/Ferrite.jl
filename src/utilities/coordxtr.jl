"""
    coordxtr(Edof,Coord,Dof,nen) -> Ex, Ey, Ez

Extracts the coordinates of the nodes of the elements.

This function can be slow for large number of elements.
"""
function coordxtr(Edof,Coord,Dof,nen)

    # TODO: Input checks

    dofsperele, nel = size(Edof)
    dofsperele -= 1 # Compensate for ele index
    ndim, nnodes = size(Coord)
    nend = div(dofsperele, nen)

    Ex = zeros(nen, nel)
    Ey = zeros(Ex)
    Ez = zeros(Ex);

    for i = 1:nel
        nodnum = zeros(Int, nen)
        for j = 1:nen
            ele_dof =  Edof[(j-1)*nend + 2 : j*nend + 1, i]
             for k = 1:nnodes
                s = zero(eltype(Edof))
                for l in 1:size(Dof, 1)
                    s += abs(Dof[l, k] - ele_dof[l])
                end
                if s == 0
                    nodnum[j] = k
                    break
                end
            end
        end

        Ex[:, i] = Coord[1, nodnum]
        if ndim > 1
            Ey[:, i] = Coord[2, nodnum]
        end
        if ndim > 2
            Ez[:, i] = Coord[3, nodnum]
        end
    end
    return Ex, Ey, Ez
end

"""
    topologyxtr(Edof,Coord,Dof,nen) -> topology

Extracts the connectivity matrix.

This function can be slow for large number of elements.
"""
function topologyxtr(Edof,Coord,Dof,nen)

    # TODO: Input checks

    dofsperele, nel = size(Edof)
    dofsperele -= 1 # Compensate for ele index
    ndim, nnodes = size(Coord)
    nend = div(dofsperele, nen)

    topology = zeros(Int, nen, nel)

    for i = 1:nel
        nodnum = zeros(Int, nen)
        for j = 1:nen
            ele_dof =  Edof[(j-1)*nend + 2 : j*nend + 1, i]
             for k = 1:nnodes
                s = zero(eltype(Edof))
                for l in 1:size(Dof, 1)
                    s += abs(Dof[l, k] - ele_dof[l])
                end
                if s == 0
                    nodnum[j] = k
                    break
                end
            end
        end

        topology[:, i] = nodnum

    end
    return topology
end
