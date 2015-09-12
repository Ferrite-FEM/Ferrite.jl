# Add docstrings

"""
Computes the stiffness matrix for a four node isoparametric
quadraterial element
"""
function plani4e(ex::VecOrMat, ey::VecOrMat,
                 ep::Array, D::Matrix, eq::VecOrMat=[0.0,0.0]) end

"""
Computes the stiffness matrix for a eight node isoparametric
quadraterial element
"""
function plani8e(ex::VecOrMat, ey::VecOrMat, ep,
                 D::Matrix, eq::VecOrMat=[0.0,0.0]) end

"""
Computes the stiffness matrix for a three node isoparametric
triangular element
"""
function plante(ex::VecOrMat, ey::VecOrMat, ep, D::Matrix, eq::VecOrMat=[0.0,0.0]) end

"""
Computes the stiffness matrix for a 8node isoparametric
hexaedric element
"""
function soli8e(ex::VecOrMat, ey::VecOrMat, ez::VecOrMat, ep, D::Matrix, eq::VecOrMat=[0.0,0.0, 0.0]) end


# Generate 2D elements
for (f_calfem, f_juafem, p_size) in ((:plani4e, :solid_square_1, 2),
                                     (:plani8e, :solid_square_2, 2),
                                     (:plante,  :solid_tri_1, 2),
                                     (:flw2i4e, :heat_square_1, 1))
    @eval begin
        function $f_calfem(ex::VecOrMat, ey::VecOrMat, ep::Array,
                           D::Matrix, eq::VecOrMat=zeros($p_size))
            # TODO, fix plane stress
            ptype = convert(Int, ep[1])
            t = ep[2]
            int_order = convert(Int, ep[3])
            x = [ex ey]
            $f_juafem(x, D, t, eq, int_order)
        end
    end
end

# Generate 3D elements
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

