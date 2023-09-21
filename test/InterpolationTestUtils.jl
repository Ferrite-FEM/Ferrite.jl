module InterpolationTestUtils
    using Ferrite
    using Test
    import LinearAlgebra: normalize
    import Random: randperm

    function find_matching_face(grid, face::FaceIndex)
        cell, facenr = face
        face_vertices = Set(Ferrite.faces(getcells(grid, cell))[facenr])
        for cnr in 1:getncells(grid)
            cnr == cell && continue
            for (i, f_vert) in enumerate(Ferrite.faces(getcells(grid, cnr)))
                face_vertices == Set(f_vert) && return FaceIndex(cnr, i)
            end
        end
        return nothing
    end
    
    function test_continuity(dh::DofHandler, face::FaceIndex;
            transformation_function::Function=identity,
            value_function::Function=function_value)
        # transformation_function: (v,n) -> z
        # Examples 
        # * Tangential continuity: fun(v, n) = v - (v ⋅ n)*n
        # * Normal continuity: fun(v, n) = v ⋅ n
        # value_function: (fe_v, q_point, ue) -> z
        
        # Check validity of input
        @assert length(dh.subdofhandlers) == 1
        @assert Ferrite.nfields(dh) == 1
    
        # Find the matching FaceIndex
        cellnr, facenr = face
        face2 = find_matching_face(dh.grid, face)
        face2 === nothing && return false
    
        # Pick "random" points on the face
        cell = getcells(dh.grid, cellnr)
        RefShape = Ferrite.getrefshape(getcells(dh.grid, cellnr))
        ip_geo = Ferrite.default_interpolation(typeof(cell))
        ip_fun = Ferrite.getfieldinterpolation(dh, (1,1))
        fqr = FaceQuadratureRule{RefShape}(10)
        fv = FaceValues(fqr, ip_fun, ip_geo)
        cell_coords = getcoordinates(dh.grid, cellnr)
        inds = randperm(getnquadpoints(fv))[1:min(4, getnquadpoints(fv))]
        
        # Random dof vector to test continuity 
        u = rand(ndofs(dh))
    
        # Calculate coordinates and function values for these
        point_coords = zeros(eltype(cell_coords), length(inds))
        point_normal = similar(point_coords)
        fun_vals = zeros(typeof(shape_value(fv, 1, 1)), length(inds))
        reinit!(fv, cell_coords, facenr, cell)
        ue = u[celldofs(dh, cellnr)]
        for (i, q_point) in enumerate(inds)
            point_coords[i] = spatial_coordinate(fv, q_point, cell_coords)
            point_normal[i] = getnormal(fv, q_point)
            fun_vals[i] = value_function(fv, q_point, ue)
        end
    
        # Calculate function values on the other cell
        cell2 = getcells(dh.grid, face2[1])
        cell_coords2 = getcoordinates(dh.grid, face2[1])
        local_coords = map(x->Ferrite.find_local_coordinate(ip_geo, cell_coords2, x), point_coords)
        @assert all(first.(local_coords)) # check that find_local_coordinate converged
        ξs = collect(last.(local_coords)) # Extract the local coordinates
        qr = QuadratureRule{RefShape}(zeros(length(ξs)), ξs)
        cv = CellValues(qr, ip_fun, ip_geo)
        reinit!(cv, cell_coords2, cell2)
        fun_vals2 = similar(fun_vals)
        ue2 = u[celldofs(dh, face2[1])]
        for q_point in 1:getnquadpoints(cv)
            @assert spatial_coordinate(cv, q_point, cell_coords2) ≈ point_coords[q_point]
            fun_vals2[q_point] = value_function(cv, q_point, ue2)
        end
    
        d1 = map((v,n)->transformation_function(v,n), fun_vals, point_normal) 
        d2 = map((v,n)->transformation_function(v,n), fun_vals2, point_normal)
        @test d1 ≈ d2
        return true
    end
    
    function create_gradcheck_qr(ip_geo::Interpolation{RefShape}, ΔL) where RefShape
        dim = Ferrite.getdim(ip_geo)
        xref = Ferrite.reference_coordinates(ip_geo)
        xc = sum(xref)/length(xref)
        ws = rand(length(xref))*((1-ΔL)/length(xref))
        xp = xc + sum(map((x,w) -> w*(x - xc), xref, ws))
        v = normalize(rand(Vec{dim}) - ones(Vec{dim})/2)
        x1 = xp + ΔL*v
        qr_w = [NaN, NaN]
        qr_x = [xp, x1]
        return QuadratureRule{RefShape}(qr_w, qr_x)
    end
    
    function test_gradient(dh, cellnr; ΔL=1e-6)
        ue = rand(ndofs_per_cell(dh, cellnr))
        x = getcoordinates(dh.grid, cellnr)
        cell = getcells(dh.grid, cellnr)
        ip_geo = Ferrite.default_interpolation(typeof(cell))
        ip_fun = Ferrite.getfieldinterpolation(dh, (1,1))
        qr = create_gradcheck_qr(ip_geo, ΔL)
        cv = CellValues(qr, ip_fun, ip_geo)
        reinit!(cv, x, cell)
        Δu_num = function_value(cv, 2, ue) - function_value(cv, 1, ue)
        Δx = spatial_coordinate(cv, 2, x) - spatial_coordinate(cv, 1, x)
        ∇u1 = function_gradient(cv, 1, ue)
        ∇u2 = function_gradient(cv, 2, ue)
        Δu_ana = 0.5*(∇u1+∇u2) ⋅ Δx
        # Δu_ana_var = 0.5*(∇u2-∇u1) ⋅ Δx # Relevant to compare magnitude if test fails
        @test Δu_num ≈ Δu_ana
        return nothing
    end

end