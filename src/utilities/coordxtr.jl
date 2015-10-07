"""
    coordxtr(Edof,Coord,Dof,nen) -> Ex, Ey, Ez

Extracts the coordinates of the nodes of the elements.

This function can be slow for large number of elements.
"""
function coordxtr(Edof,Coord,Dof,nen)

    # TODO: Input checks

    nel, dofsperele = size(Edof)
    dofsperele -= 1 # Compensate for ele index
    nnodes, ndim = size(Coord)
    nend = div(dofsperele, nen)

    Ex = zeros(nel, nen)
    Ey = zeros(Ex)
    Ez = zeros(Ex);

    for i = 1:nel
        nodnum = zeros(Int, nen)
        for j = 1:nen
            ele_dof =  Edof[i, (j-1)*nend + 2 : j*nend + 1]
            for k = 1:nnodes
                if Dof[k, :] - ele_dof == [0 0]
                    nodnum[j] = k
                    break
                end
            end
        end

        Ex[i, :] = Coord[nodnum, 1]
        if ndim > 1
            Ey[i, :] = Coord[nodnum, 2]
        end
        if ndim > 2
            Ez[i, :] = Coord[nodnum, 3]
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

    nel, dofsperele = size(Edof)
    dofsperele -= 1 # Compensate for ele index
    nnodes, ndim = size(Coord)
    nend = div(dofsperele, nen)

    topology = zeros(Int, nel, nen)

    for i = 1:nel
        nodnum = zeros(Int, nen)
        for j = 1:nen
            ele_dof =  Edof[i, (j-1)*nend + 2 : j*nend + 1]
            for k = 1:nnodes
                if Dof[k, :] - ele_dof == [0 0]
                    nodnum[j] = k
                    break
                end
            end
        end

        topology[i, :] = nodnum

    end
    return topology
end
