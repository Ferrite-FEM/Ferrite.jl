@testset "apply_analytical!" begin

    # Convenience helper functions
    typebase(T::DataType) = T.name.wrapper
    typebase(v) = typebase(typeof(v))
    change_ip_order(ip::Interpolation, ::Nothing) = ip
    function change_ip_order(ip::Interpolation, order::Int)
        B = typebase(ip)
        Dim = Ferrite.getdim(ip)
        RefShape = Ferrite.getrefshape(ip)
        return B{Dim,RefShape,order}()
    end
    getcellorder(CT) = Ferrite.getorder(Ferrite.default_interpolation(CT))
    getcelltypedim(::Type{<:Cell{dim}}) where dim = dim

    # Functions to create dof handlers for testing
    function testdh(CT, ip_order_u, ip_order_p)
        dim = getcelltypedim(CT)
        local grid
        try
            grid = generate_grid(CT, ntuple(_->3, dim))
        catch e
            isa(e, MethodError) && e.f==generate_grid && return nothing
            rethrow(e)
        end

        dh = DofHandler(grid)
        default_ip = Ferrite.default_interpolation(CT)
        try
            push!(dh, :u, dim, change_ip_order(default_ip, ip_order_u))
            push!(dh, :p, 1, change_ip_order(default_ip, ip_order_p))                
        catch e
            isa(e, MethodError) && e.f == Ferrite.reference_coordinates && return nothing
            rethrow(e)
        end
        close!(dh)
        return dh
    end

    function testmdh(dim, ip_order_u, ip_order_p)
        if dim == 1
            nodes = Node.([Vec{1}((x,)) for x in 0.0:1.0:3.0])
            cell1 = Cell{1,2,2}((1,2))
            cell2 = Cell{1,3,2}((2,3,4))
            grid = Grid([cell1, cell2], nodes; cellsets=Dict("A"=>Set(1:1), "B"=>Set(2:2)))
        elseif dim == 2
            nodes = Node.([Vec{2}((x,y)) for y in 0.0:2 for x in 0.0:1])
            cell1 = Cell{2,3,3}((1,2,3))
            cell2 = Cell{2,3,3}((2,4,3))
            cell3 = Cell{2,4,4}((3,4,6,5))
            grid = Grid([cell1,cell2,cell3], nodes; cellsets=Dict("A"=>Set(1:2), "B"=>Set(3:3)))
        else
            error("Only dim=1 & 2 supported")
        end
        mdh = MixedDofHandler(grid)
        default_ip_A = Ferrite.default_interpolation(getcelltype(grid, first(getcellset(grid,"A"))))
        default_ip_B = Ferrite.default_interpolation(getcelltype(grid, first(getcellset(grid,"B"))))
        ufield_A = Field(:u, change_ip_order(default_ip_A, ip_order_u), dim)
        pfield_A = Field(:p, change_ip_order(default_ip_A, ip_order_p), 1)
        ufield_B = Field(:u, change_ip_order(default_ip_B, ip_order_u), dim)
        push!(mdh, FieldHandler([ufield_A, pfield_A], getcellset(grid,"A")))
        push!(mdh, FieldHandler([ufield_B,], getcellset(grid, "B")))
        close!(mdh)
        return mdh
    end

    # The following can be removed after #457 is merged if that will include the MixedDofHandler
    function _global_dof_range(dh::MixedDofHandler, field_name::Symbol)
        dofs = Set{Int}()
        for fh in dh.fieldhandlers
            if field_name ∈ Ferrite.getfieldnames(fh)
                _global_dof_range!(dofs, dh, fh, field_name, fh.cellset)
            end
        end
        return sort!(collect(Int, dofs))
    end

    function _global_dof_range(dh::DofHandler, field_name::Symbol)
        dofs = Set{Int}()
        _global_dof_range!(dofs, dh, dh, field_name, 1:getncells(dh.grid))
        return sort!(collect(Int, dofs))
    end

    function _global_dof_range!(dofs, dh, dh_fh, field_name, cellset)
        eldofs = celldofs(dh, first(cellset))
        field_range = dof_range(dh_fh, field_name)
        for i in cellset
            celldofs!(eldofs, dh, i)
            for j in field_range
                @inbounds d = eldofs[j]
                d in dofs || push!(dofs, d)
            end
        end
    end

    @testset "DofHandler" begin
        for (CT,name) in Ferrite.celltypes
            for ip_order_u in 1:2
                for ip_order_p in 1:2
                    dh = testdh(CT, ip_order_u, ip_order_p)
                    isnothing(dh) && continue # generate_grid not supported for this CT, or reference_coordinates not defined
                    dim = Ferrite.getdim(dh.grid)
                    num_udofs = length(_global_dof_range(dh, :u))
                    num_pdofs = length(_global_dof_range(dh, :p))

                    # Test average value
                    a = zeros(ndofs(dh))
                    f(x) = ones(Vec{dim})
                    apply_analytical!(a, dh, :u, f)
                    @test sum(a)/length(a) ≈ num_udofs/(num_udofs+num_pdofs)

                    # If not super/subparametric, compare with ConstraintHandler and node set 
                    if ip_order_u==ip_order_p==getcellorder(CT)    
                        fill!(a, 0)
                        a_ch = copy(a)
                        fp(x) = norm(x)^2
                        fu(x::Vec{1}) = Vec{1}((sin(x[1]),))
                        fu(x::Vec{2}) = Vec{2}((sin(x[1]), cos(x[2])))
                        fu(x::Vec{3}) = Vec{3}((sin(x[1]), cos(x[2]), x[3]*(x[1]+x[2])))
                        ch = ConstraintHandler(dh)
                        add!(ch, Dirichlet(:p, Set(1:getnnodes(dh.grid)), (x,t)->fp(x)))
                        add!(ch, Dirichlet(:u, Set(1:getnnodes(dh.grid)), (x,t)->fu(x)))
                        close!(ch); update!(ch, 0.0)
                        apply!(a_ch, ch)

                        apply_analytical!(a, dh, :u, fu)
                        apply_analytical!(a, dh, :p, fp)

                        @test a ≈ a_ch 
                    end 
                end
            end
        end
    end

    @testset "MixedDofHandler" begin
        for dim in 1:2
            for ip_order_u in 1:2
                for ip_order_p in 1:2
                    dh = testmdh(dim, ip_order_u, ip_order_p)
                    num_udofs = length(_global_dof_range(dh, :u))
                    num_pdofs = length(_global_dof_range(dh, :p))

                    # Test average value
                    a = zeros(ndofs(dh))
                    f(x) = ones(Vec{dim})
                    apply_analytical!(a, dh, :u, f)
                    @test sum(a)/length(a) ≈ num_udofs/(num_udofs+num_pdofs)
                end
            end
        end
    end

    @testset "Exceptions" begin
        dh = testdh(Quadrilateral, 1, 1)
        @test_throws ErrorException apply_analytical!(zeros(ndofs(dh)), dh, :v, x->0.0)    # Missing field
        @test_throws ErrorException apply_analytical!(zeros(ndofs(dh)), dh, :u, x->0.0)    # Should be f(x)::Vec{2}

        mdh = testmdh(2, 1, 1)
        @test_throws ErrorException apply_analytical!(zeros(ndofs(mdh)), mdh, :v, x->0.0)  # Missing field
        @test_throws ErrorException apply_analytical!(zeros(ndofs(mdh)), mdh, :u, x->0.0)  # Should be f(x)::Vec{2}
    end
end
