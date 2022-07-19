# If face_value is ScalarValued, f should return scalar 
# If face_value is VectorValued, f should return a vector
struct Neumann{FV,FT}
    field_name::Symbol
    face_values::FV
    faces::Vector{Int}
    cells::Vector{Int}
    fun::FT # f(x::Vec, t::Number, normal::Vec)->Union{<:Number, <:Vec}
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
        calculate_neumann_contribution!(fe, face, nbc.face_values, time, nbc.fun)
        assemble!(f, view(celldofs(face), dofs), fe)
    end
end

function calculate_neumann_contribution!(fe::Vector, face::FaceIterator, fv::FaceValues, time, fun)
    fill!(fe, 0)
    reinit!(fv, face)
    for q_point in 1:getnquadpoints(fv)
        x = spatial_coordinate(fv, q_point, getcoordinates(face))
        for i in 1:getnbasefunctions(fv)
            fe[i] += fun(fv, q_point, i, x, time)
        end
    end
end