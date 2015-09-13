# Add docstrings

################
# Solid elements
################
"""
Computes the stiffness matrix and force vector for a
four node isoparametric quadraterial element
"""
function plani4e(ex::VecOrMat, ey::VecOrMat,
                 ep::Array, D::Matrix, eq::VecOrMat=[0.0,0.0]) end
"""
Computes the stiffness matrix and force vector for an eight node isoparametric quadraterial element
"""
function plani8e(ex::VecOrMat, ey::VecOrMat, ep,
                 D::Matrix, eq::VecOrMat=[0.0,0.0]) end

"""
Computes the stiffness matrix and force vector for a three node isoparametric
triangular element
"""
function plante(ex::VecOrMat, ey::VecOrMat, ep, D::Matrix, eq::VecOrMat=[0.0,0.0]) end

"""
Computes the stiffness matrix and force vector for a
eight node isoparametric hexahedron element.
"""
function soli8e(ex::VecOrMat, ey::VecOrMat, ez::VecOrMat, ep, D::Matrix, eq::VecOrMat=[0.0,0.0,0.0]) end


################
# Heat elements
################
"""
Computes the stiffness matrix and force vector for a
four node isoparametric quadraterial heat transfer element
"""
function flw2i4e(ex::VecOrMat, ey::VecOrMat,
                 ep::Array, D::Matrix, eq::VecOrMat=[0.0]) end
"""
Computes the stiffness matrix and force vector for an
eight node isoparametric quadraterial element
"""
function flw2i8e(ex::VecOrMat, ey::VecOrMat, ep,
                 D::Matrix, eq::VecOrMat=[0.0]) end

"""
Computes the stiffness matrix and force vector for a three node isoparametric
triangular heat transfer element
"""
function flw2te(ex::VecOrMat, ey::VecOrMat, ep, D::Matrix, eq::VecOrMat=[0.0]) end

"""
Computes the stiffness matrix and force vector for a
eight node isoparametric hexahedron heat transfer  element.
"""
function flw3i8e(ex::VecOrMat, ey::VecOrMat, ez::VecOrMat, ep, D::Matrix, eq::VecOrMat=[0.0]) end


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



