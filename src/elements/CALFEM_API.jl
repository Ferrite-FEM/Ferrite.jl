# Generate 2D solid elements
for (f_calfem, f_juafem) in ((:plani4e, :solid_square_1e),
                             (:plani8e, :solid_square_2e),
                             (:plante,  :solid_tri_1e),
                             (:plani4s, :solid_square_1s),
                             (:plani8s, :solid_square_2s),
                             (:plants,  :solid_tri_1s))
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
for (f_calfem, f_juafem) in ((:soli8e, :solid_cube_1e),
                             (:soli8s, :solid_cube_1s))
    @eval begin
        function $f_calfem(ex::VecOrMat, ey::VecOrMat, ez::VecOrMat, ep::Array,
                           D::Matrix, eq::VecOrMat=zeros(3))
            int_order = convert(Int, ep[1])
            x = [ex ey ez]
            $f_juafem(x, D, eq, int_order)
        end
    end
end

# Generate 2D heat elements
for (f_calfem, f_juafem) in ((:flw2i4e, :heat_square_1e),
                             (:flw2i8e, :heat_square_2e),
                             (:flw2te,  :heat_tri_1e),
                             (:flw2i4s, :heat_square_1s),
                             (:flw2i8s, :heat_square_2s),
                             (:flw2ts,  :heat_tri_1s))
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
for (f_calfem, f_juafem) in ((:flw3i8e, :heat_cube_1e),
                             (:flw3i8s, :heat_cube_1s))
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
