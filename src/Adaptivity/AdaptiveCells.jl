abstract type AbstractAdaptiveGrid{dim} <: AbstractGrid{dim} end
abstract type AbstractAdaptiveCell{dim,N,M} <: AbstractCell{dim,N,M} end

_maxlevel = [30,19]

function set_maxlevel(dim::Integer,maxlevel::Integer)
    _maxlevel[dim-1] = maxlevel
end

struct OctantBWG{dim, N, M} <: AbstractCell{dim,N,M}
    #Refinement level
    l::UInt
    #x,y,z \in {0,...,2^b} where (0 ≤ l ≤ b)}
    xyz::NTuple{dim,Int} 
end

"""
    OctantBWG(dim::Integer, l::Integer, b::Integer, m::Integer)
Construct an `octant` based on dimension `dim`, level `l`, amount of levels `b` and morton index `m`
"""
function OctantBWG(dim::Integer, l::Integer, m::Integer, b::Integer=_maxlevel[dim-1])
    @assert m ≤ 2^(dim*l)
    x,y,z = (0,0,0) 
    h = _compute_size(b,l) 
    for i in 0:l-1
        x = x | (h*((m-1) & 2^(dim*i))÷2^((dim-1)*i))
        y = y | (h*((m-1) & 2^(dim*i+1))÷2^((dim-1)*i+1))
        z = z | (h*((m-1) & 2^(dim*i+2))÷2^((dim-1)*i+2))
    end
    if dim == 2
        OctantBWG{dim,4,4}(l,(x,y)) 
    elseif dim == 3
        OctantBWG{dim,8,6}(l,(x,y,z)) 
    else
        error("$dim Dimension not supported")
    end 
end

Base.zero(::Type{OctantBWG{3, 8, 6}}) = OctantBWG(3, 0, 1)
Base.zero(::Type{OctantBWG{2, 4, 4}}) = OctantBWG(2, 0, 1)

# Follow z order, x before y before z for faces, edges and corners
struct OctreeBWG{dim,N,M} <: AbstractAdaptiveCell{dim,N,M}
    leaves::Vector{OctantBWG{dim,N,M}}
    #maximum refinement level 
    b::UInt
    nodes::NTuple{N,Int}
end

OctreeBWG{3,8,6}(nodes::NTuple,b=_maxlevel[2]) = OctreeBWG{3,8,6}([zero(OctantBWG{3,8,6})],b,nodes)
OctreeBWG{2,4,4}(nodes::NTuple,b=_maxlevel[1]) = OctreeBWG{2,4,4}([zero(OctantBWG{2,4,4})],b,nodes)
OctreeBWG(cell::Quadrilateral) = OctreeBWG{2,4,4}(cell.nodes)
OctreeBWG(cell::Hexahedron) = OctreeBWG{3,8,6}(cell.nodes)

Base.length(tree::OctreeBWG) = length(tree.leaves)

"""
    ForestBWG{dim, C<:AbstractAdaptiveCell, T<:Real} <: AbstractAdaptiveGrid{dim}
`p4est` adaptive grid implementation based on Burstedde, Wilcox, Ghattas [2011]
"""
struct ForestBWG{dim, C<:OctreeBWG, T<:Real} <: AbstractAdaptiveGrid{dim}
    cells::Vector{C}
    nodes::Vector{Node{dim,T}}
    # Sets
    cellsets::Dict{String,Set{Int}}
    nodesets::Dict{String,Set{Int}}
    facesets::Dict{String,Set{FaceIndex}} 
    edgesets::Dict{String,Set{EdgeIndex}} 
    vertexsets::Dict{String,Set{VertexIndex}}
    #Topology
    topology::ExclusiveTopology
end

function ForestBWG(grid::AbstractGrid{dim}) where dim
    cells = getcells(grid)
    C = eltype(cells)
    @assert isconcretetype(C)
    @assert (C == Quadrilateral && dim == 2) || (C == Hexahedron && dim == 3)
    topology = ExclusiveTopology(cells)
    cells = OctreeBWG.(grid.cells)
    nodes = getnodes(grid)
    cellsets = getcellsets(grid)
    nodesets = getnodesets(grid)
    facesets = getfacesets(grid)
    edgesets = getedgesets(grid)
    vertexsets = getvertexsets(grid)
    return ForestBWG(cells,nodes,cellsets,nodesets,facesets,edgesets,vertexsets,topology)
end

function getncells(grid::ForestBWG)
    numcells = 0
    for tree in grid.cells
        numcells += length(tree) 
    end
    return numcells
end

function Base.show(io::IO, ::MIME"text/plain", agrid::ForestBWG)
    println(io, "ForestBWG with ")
    println(io, "   $(getncells(agrid)) cells")
end

"""
    child_id(octant::OctantBWG, b::Integer)
Given some OctantBWG `octant` and maximum refinement level `b`, compute the child_id of `octant`
note the following quote from Burstedde et al:
  children are numbered from 0 for the front lower left child, 
  to 1 for the front lower right child, to 2 for the back lower left, and so on, with
  4, . . . , 7 being the four children on top of the children 0, . . . , 3.
shifted by 1 due to julia 1 based indexing 
"""
function child_id(octant::OctantBWG{3},b::Integer=_maxlevel[2])
    i = 0x00
    h = _compute_size(b,octant.l)
    x,y,z = octant.xyz
    i = i | ((x & h) != 0x00 ? 0x01 : 0x00)
    i = i | ((y & h) != 0x00 ? 0x02 : 0x00)
    i = i | ((z & h) != 0x00 ? 0x04 : 0x00)
    return i+0x01
end

function child_id(octant::OctantBWG{2},b::Integer=_maxlevel[1])
    i = 0x00
    h = _compute_size(b, octant.l)
    x,y = octant.xyz
    i = i | ((x & h) != 0x00 ? 0x01 : 0x00)
    i = i | ((y & h) != 0x00 ? 0x02 : 0x00)
    return i+0x01
end

function parent(octant::OctantBWG{dim,N,M}, b::Integer=_maxlevel[dim-1]) where {dim,N,M}
    if octant.l > 0 
        h = _compute_size(b,octant.l)
        l = octant.l - 0x01
        return OctantBWG{dim,N,M}(l,octant.xyz .& ~h)
    else 
        error("root has no parent")
    end
end

"""
    descendants(octant::OctantBWG, b::Integer)
Given an `octant`, computes the two smallest possible octants that fit into the first and last corners
of `octant`, respectively. These computed octants are called first and last descendants of `octant`
since they are connected to `octant` by a path down the octree to the maximum level  `b`
"""
function descendants(octant::OctantBWG{dim,N,M}, b::Integer=_maxlevel[dim-1]) where {dim,N,M}
    l1 = b-1; l2 = b-1 # not sure 
    h = _compute_size(b,octant.l)
    return OctantBWG{dim,N,M}(l1,octant.xyz), OctantBWG{dim,N,M}(l2,octant.xyz .+ (h-2))
end

function face_neighbor(octant::OctantBWG{dim,N,M}, f::Integer, b::Integer=_maxlevel[dim-1]) where {dim,N,M}
    l = octant.l
    h = _compute_size(b,octant.l)
    x,y,z = octant.xyz 
    x += ((f == 1) ? -h : ((f == 2) ? h : 0))
    y += ((f == 3) ? -h : ((f == 4) ? h : 0))
    z += ((f == 5) ? -h : ((f == 6) ? h : 0))
    return OctantBWG{dim,N,M}(l,(x,y,z))
end

"""
    edge_neighbor(octant::OctantBWG, e::Integer, b::Integer)
Computes the edge neighbor octant which is only connected by the edge `e` to `octant`
"""
function edge_neighbor(octant::OctantBWG{3,N,M}, e::Integer, b::Integer=_maxlevel[2]) where {N,M}
    @assert 1 ≤ e ≤ 12
    e -= 1
    l = octant.l
    h = _compute_size(b,octant.l)
    ox,oy,oz = octant.xyz
    case = e ÷ 4
    if case == 0
        x = ox 
        y = oy + (2*(e & 0x01) - 1)*h 
        z = oz + ((e & 0x02) - 1)*h
        return OctantBWG{3,N,M}(l,(x,y,z))
    elseif case == 1
        x = ox  + (2*(e & 0x01) - 1)*h 
        y = oy 
        z = oz + ((e & 0x02) - 1)*h
        return OctantBWG{3,N,M}(l,(x,y,z))  
    elseif case == 2
        x = ox + (2*(e & 0x01) - 1)*h 
        y = oy + ((e & 0x02) - 1)*h
        z = oz
        return OctantBWG{3,N,M}(l,(x,y,z))
    else
        error("edge case not found")
    end
end

"""
    corner_neighbor(octant::OctantBWG, c::Integer, b::Integer)
Computes the corner neighbor octant which is only connected by the corner `c` to `octant`
"""
function corner_neighbor(octant::OctantBWG{3,N,M}, c::Integer, b::Integer=_maxlevel[2]) where {N,M}
    c -= 1
    l = octant.l
    h = _compute_size(b,octant.l)
    ox,oy,oz = octant.xyz
    x = ox + (2*(c & 1) - 1)*h 
    y = oy + ((c & 2) - 1)*h
    z = oz + ((c & 4)/2 - 1)*h
    return OctantBWG{3,N,M}(l,(x,y,z))
end

function corner_neighbor(octant::OctantBWG{2,N,M}, c::Integer, b::Integer=_maxlevel[1]) where {N,M}
    c -= 1
    l = octant.l
    h = _compute_size(b,octant.l)
    ox,oy = octant.xyz
    x = ox + (2*(c & 1) - 1)*h 
    y = oy + ((c & 2) - 1)*h
    return OctantBWG{2,N,M}(l,(x,y))
end

function Base.show(io::IO, ::MIME"text/plain", o::OctantBWG{3,N,M}) where {N,M}
    x,y,z = o.xyz
    println(io, "OctantBWG{3,$N,$M}")
    println(io, "   l = $(o.l)")
    println(io, "   xyz = $x,$y,$z")
end

function Base.show(io::IO, ::MIME"text/plain", o::OctantBWG{2,N,M}) where {N,M}
    x,y = o.xyz
    println(io, "OctantBWG{2,$N,$M}")
    println(io, "   l = $(o.l)")
    println(io, "   xy = $x,$y")
end

_compute_size(b::Integer,l::Integer) = 2^(b-l)
# return the two adjacent faces $f_i$ adjacent to edge `edge`
_face(edge::Int) = 𝒮[edge, :]
# return the `i`-th adjacent face fᵢ to edge `edge`
_face(edge::Int, i::Int) = 𝒮[edge, i]
# return two face corners ξᵢ of the face `face` along edge `edge`
_face_edge_corners(edge::Int, face::Int) = 𝒯[edge,face] 
# return the two `edge` corners cᵢ
_edge_corners(edge::Int) = 𝒰[edge,:]
# return the `i`-th edge corner of `edge`
_edge_corners(edge::Int,i::Int) = 𝒰[edge,i]
# finds face corner ξ′ in f′ for two associated faces f,f′ in {1,...,6} and their orientation r in {1,...,4}}
_neighbor_corner(f::Int,f′::Int,r::Int,ξ::Int) = 𝒫[𝒬[ℛ[f,f′],r],ξ]

# map given `face` and `ξ` to corner `c`. Need to provide dim for different lookup 
function _face_corners(dim::Int,face::Int,ξ::Int)
    if dim == 2
        return 𝒱₂[face,ξ] 
    elseif dim == 3
        return 𝒱₃[face,ξ]
    else
        error("No corner-lookup table available")
    end
end

function _face_corners(dim::Int,face::Int)
    if dim == 2
        return 𝒱₂[face,:] 
    elseif dim == 3
        return 𝒱₃[face,:]
    else
        error("No corner-lookup table available")
    end
end

##### OCTANT LOOK UP TABLES ######
const 𝒮 = [3  5
           4  5
           3  6
           4  6
           1  5
           2  5
           1  6
           2  6
           1  3
           2  3
           1  4
           2  4] 

# (0,0) non existing connections
const 𝒯 = [(0, 0)  (0, 0)  (1, 2)  (0, 0)  (1, 2)  (0, 0)
           (0, 0)  (0, 0)  (0, 0)  (1, 2)  (3, 4)  (0, 0)
           (0, 0)  (0, 0)  (3, 4)  (0, 0)  (0, 0)  (1, 2)
           (0, 0)  (0, 0)  (0, 0)  (3, 4)  (0, 0)  (3, 4)
           (1, 2)  (0, 0)  (0, 0)  (0, 0)  (1, 3)  (0, 0)
           (0, 0)  (1, 2)  (0, 0)  (0, 0)  (2, 4)  (0, 0)
           (3, 4)  (0, 0)  (0, 0)  (0, 0)  (0, 0)  (1, 3)
           (0, 0)  (3, 4)  (0, 0)  (0, 0)  (0, 0)  (2, 4)
           (1, 3)  (0, 0)  (1, 3)  (0, 0)  (0, 0)  (0, 0)
           (0, 0)  (1, 3)  (2, 4)  (0, 0)  (0, 0)  (0, 0)
           (2, 4)  (0, 0)  (0, 0)  (1, 3)  (0, 0)  (0, 0)
           (0, 0)  (2, 4)  (0, 0)  (2, 4)  (0, 0)  (0, 0)]

const 𝒰 = [1  2
           3  4
           5  6
           7  8
           1  3
           2  4
           5  7
           6  8
           1  5
           2  6
           3  7
           4  8]

const 𝒱₂ = [1  3
            2  4
            1  2
            3  4] 

const 𝒱₃ = [1  3  5  7
            2  4  6  8
            1  2  5  6
            3  4  7  8
            1  2  3  4
            5  6  7  8]

const ℛ = [1  2  2  1  1  2
           3  1  1  2  2  1
           3  1  1  2  2  1
           1  3  3  1  1  2
           1  3  3  1  1  2
           3  1  1  3  3  1]

const 𝒬 = [2  3  6  7
           1  4  5  8
           1  5  4  8]

const 𝒫 = [1  2  3  4
           1  3  2  4
           2  1  4  3
           2  4  1  3
           3  1  4  2
           3  4  1  2
           4  2  3  1
           4  3  2  1]
