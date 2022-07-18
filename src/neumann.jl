# If face_value is ScalarValued, f should return scalar 
# If face_value is VectorValued, f should return a vector
struct Neumann{FV}      # cells and faces are a deconstructed
    cells::Vector{Int}  # FaceIndex, to allow loop over cells 
    faces::Vector{Int}  # with CellIterator 
    face_values::FV     
    field_name::Symbol
    fun # f(x::Vec, t::Number, normal::Vec)->Union{<:Number, <:Vec}
end

function Neumann(field_name::Symbol, ∂Ω::Set{FaceIndex}, fv::FaceValues, f)
    cells, faces = tuple((collect([face[j] for face in ∂Ω]) for j in 1:2)...)
    return Neumann(cells, faces, fv, field_name, f)
end

function apply!(f::Vector{T}, nbc::Neumann, dh::DofHandler, time) where T
    dofs = collect(dof_range(dh, nbc.field_name))
    fe = zeros(T, length(dofs))
    for (i, cell) in enumerate(CellIterator(dh, nbc.cells))
        calculate_element_force!(fe, cell, nbc.face_values, nbc.faces[i], time, nbc.fun)
        assemble!(f, view(celldofs(cell), dofs), fe)
    end
end

function calculate_element_force!(fe::Vector, cell::CellIterator, fv::FaceValues, face_nr::Int, time, fun)
    fill!(fe, 0)
    reinit!(fv, cell, face_nr)
    for q_point in 1:getnquadpoints(fv)
        dΓ = getdetJdV(fv, q_point)
        x = spatial_coordinate(fv, q_point, cell.coords)
        n = getnormal(fv, q_point)
        b = fun(x, time, n)
        for i in 1:getnbasefunctions(fv)
            δu = shape_value(fv, q_point, i)
            fe[i] += δu ⋅ b * dΓ
        end
    end
end


struct NeumannHandler{DH<:AbstractDofHandler}
    nbc::Vector{Neumann}
    dh::DH
end

NeumannHandler(dh::AbstractDofHandler) = NeumannHandler(Neumann[], dh)

function add!(nh::NeumannHandler, nbc::Neumann)
    # Should make some checks here...
    push!(nh.nbc, nbc)
end

function apply!(f::Vector, nh::NeumannHandler, time)
    for nbc in nh.nbc
        apply!(f, nbc, nh.dh, time)
    end
end

