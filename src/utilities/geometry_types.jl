abstract Shape

immutable Dim{T} end

immutable Triangle <: Shape end
immutable Square <: Shape end


immutable MeshNode{dim, T}
    n::Vec{dim, T}
end

get_coordinates(n::MeshNode) = n.n
get_id(n::MeshNode) = n.id

immutable MeshEdge
    nodes::NTuple{2, Int}
end

abstract MeshPolygon

immutable MeshTriangle <: MeshPolygon
    edges::NTuple{3, Int}
end

immutable MeshQuadraterial <: MeshPolygon
    edges::NTuple{4, Int}
end

immutable MeshSolid{N}
    polygons::NTuple{N, Int}
end