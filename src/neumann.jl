"""
    Neumann(field_name::Symbol, fv::FaceValues, faceset::Set{FaceIndex}, f, cellset=nothing)

    Specify a Neumann boundary condition for the field `field_name` on the 
    faces in `faceset`, using the facevalues `fv`. If `isa(cellset, Vector{Int})`, 
    use only the cells in `faceset` that are in `cellset`. 
    The key to specifying the Neumann boundary condition is the function 
    `f`, which has the signature,
    ``` 
    f(fv::FaceValues, q_point::Int, i_shape::Int, x::Vec, time)
    ```
    and calculates the contribution to the external force component 
    `i_shape` from the current cell from quadrature point `q_point` 
    (with position `x`) at time `time`. 
    
    # Example
    For many cases, the weak form will have the following forms,
    ```math
    \\int_{\\Gamma} b  \\delta u \\ \\mathrm{d}\\Gamma

    \\int_{\\Gamma} \\boldsymbol{b} \\cdot \\boldsymbol{\\delta u} \\ \\mathrm{d}\\Gamma
    ```
    for scalar and vector fields, respectively. Here ``b`` or 
    ``\\boldsymbol{b}`` is the prescribed Neumann value. To avoid code 
    duplication, a standard base function can be defined as
    ```
    function standard_neumann_fun(bfun, fv::FaceValues, q_point::Int, i::Int, x::Vec, time)
        δu = shape_value(fv, q_point, i)
        b = bfun(x, time, getnormal(fv, q_point))
        dΓ = getdetJdV(fv, q_point)
        return (δu ⋅ b) * dΓ
    end
    ```
    allowing the `bfun` to be used to specialize for different 
    boundary conditions, e.g.
    ```
    function bfun(x::Vec, time, n::Vec)
        # Calculate the Neumann value for given position, time and normal vector
    end
    ```
    Note that for scalar fields, `fv::FaceScalarValues` and `bfun` should 
    return a scalar. For vector fields, `fv::FaceVectorValues` and `bfun` 
    should return a Vec with the same dimension as the field. 
    Finally, `f` can then be specified as 
    ```
    f(args...) = standard_neumann_fun(bfun, args...)
    ```
    and given to the `Neumann` boundary condition. If the weak form 
    contains multiple Neumann boundary conditions (e.g. multiple fields), 
    these can be added independently. 
    Adding multiple boundary conditions that affect the same degrees of 
    freedom, will just add up the contributions. 
"""
struct Neumann{FV,FUN}
    field_name::Symbol
    face_values::FV
    faces::Vector{Int}
    cells::Vector{Int}
    f::FUN # f(fv::FaceValues, q_point::Int, i_shape::Int, x::Vec, t::Number)->Number
end

function Neumann(field_name::Symbol, fv::FaceValues, faceset::Set{FaceIndex}, f, cellset=nothing)
    cells, faces = _get_cells_and_faces(faceset, cellset)
    return Neumann(field_name, fv, faces, cells, f)
end

struct NeumannHandler{DH<:AbstractDofHandler}
    nbc::Vector{Neumann}
    dh::DH
end

NeumannHandler(dh::AbstractDofHandler) = NeumannHandler(Neumann[], dh)

function Ferrite.add!(nh::NeumannHandler, nbc::Neumann)
    # Should make some checks here...
    push!(nh.nbc, nbc)
end

function Ferrite.apply!(f::Vector, nh::NeumannHandler, time)
    foreach(nbc->apply!(f,nbc,nh.dh,time), nh.nbc)
end

function Ferrite.apply!(f::Vector{T}, nbc::Neumann, dh::DofHandler, time) where T
    dofs = collect(dof_range(dh, nbc.field_name))
    fe = zeros(T, length(dofs))
    for face in FaceIterator(dh, nbc.faces, nbc.cells)
        calculate_neumann_contribution!(fe, face, nbc.face_values, time, nbc.f)
        assemble!(f, view(celldofs(face), dofs), fe)
    end
end

function calculate_neumann_contribution!(fe::Vector, face::FaceIterator, fv::FaceValues, time, f)
    fill!(fe, 0)
    reinit!(fv, face)
    for q_point in 1:getnquadpoints(fv)
        x = spatial_coordinate(fv, q_point, getcoordinates(face))
        for i in 1:getnbasefunctions(fv)
            fe[i] += f(fv, q_point, i, x, time)
        end
    end
end