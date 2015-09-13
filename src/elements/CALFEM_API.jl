# Generate 2D solid elements
for (f_calfem, f_juafem) in ((:plani4e, :solid_square_1),
                             (:plani8e, :solid_square_2),
                             (:plante,  :solid_tri_1))
    @eval begin
        function $f_calfem(ex::VecOrMat, ey::VecOrMat, ep::Array,
                           D::Matrix, eq::VecOrMat=zeros(2))
            # TODO, fix plane stress
            ptype = convert(Int, ep[1])
            t = ep[2]
            int_order = convert(Int, ep[3])
            x = [ex ey]
            $f_juafem(x, D, t, eq, int_order)
        end
    end
end


# Generate 3D solid elements
for (f_calfem, f_juafem) in ((:soli8e, :solid_cube_1),)
    @eval begin
        function $f_calfem(ex::VecOrMat, ey::VecOrMat, ez::VecOrMat, ep::Array,
                           D::Matrix, eq::VecOrMat=[0.0,0.0])
            int_order = convert(Int, ep[1])
            x = [ex ey ez]
            $f_juafem(x, D, eq, int_order)
        end
    end
end

# Generate 2D heat elements
for (f_calfem, f_juafem) in ((:flw2i4e, :heat_square_1),
                             (:flw2i8e, :heat_square_2),
                             (:flw2te,  :heat_tri_1))
    @eval begin
        function $f_calfem(ex::VecOrMat, ey::VecOrMat, ep::Array,
                           D::Matrix, eq::VecOrMat=zeros(1))
            # TODO, fix plane stress
            t = ep[1]
            int_order = convert(Int, ep[2])
            x = [ex ey]
            $f_juafem(x, D, t, eq, int_order)
        end
    end
end

# Generate 3D heat elements
for (f_calfem, f_juafem) in ((:flw3i8e, :heat_cube_1),)
    @eval begin
        function $f_calfem(ex::VecOrMat, ey::VecOrMat, ez::VecOrMat, ep::Array,
                           D::Matrix, eq::VecOrMat=zeros(1))
            # TODO, fix plane stress
            int_order = convert(Int, ep[1])
            x = [ex ey ez]
            $f_juafem(x, D, eq, int_order)
        end
    end
end
