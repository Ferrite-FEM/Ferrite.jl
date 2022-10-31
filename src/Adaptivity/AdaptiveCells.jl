abstract type AbstractAdaptiveGrid{dim} <: AbstractGrid{dim} end
abstract type AbstractAdaptiveCell{dim,N,M} <: AbstractCell{dim,N,M} end

_maxlevel = [30,19]

function set_maxlevel(dim::Integer,maxlevel::Integer)
    _maxlevel[dim-1] = maxlevel
end

struct OctantBWG{dim, N, M, T} <: AbstractCell{dim,N,M}
    #Refinement level
    l::T
    #x,y,z \in {0,...,2^b} where (0 ≤ l ≤ b)}
    xyz::NTuple{dim,T}
end

"""
    OctantBWG(dim::Integer, l::Integer, b::Integer, m::Integer)
Construct an `octant` based on dimension `dim`, level `l`, amount of levels `b` and morton index `m`
"""
function OctantBWG(dim::Integer, l::T, m::T, b::T=_maxlevel[dim-1]) where T <: Integer
    @assert l ≤ b #maximum refinement level exceeded
    @assert m ≤ (one(T)+one(T))^(dim*l)
    x,y,z = (zero(T),zero(T),zero(T))
    h = Int32(_compute_size(b,l))
    _zero = zero(T)
    _one = one(T)
    _two = _one + _one
    for i in _zero:l-_one
        x = x | (h*((m-_one) & _two^(dim*i))÷_two^((dim-_one)*i))
        y = y | (h*((m-_one) & _two^(dim*i+_one))÷_two^((dim-_one)*i+_one))
        z = z | (h*((m-_one) & _two^(dim*i+_two))÷_two^((dim-_one)*i+_two))
    end
    if dim == 2
        OctantBWG{dim,4,4,T}(l,(x,y))
    elseif dim == 3
        OctantBWG{dim,8,6,T}(l,(x,y,z))
    else
        error("$dim Dimension not supported")
    end
end

OctantBWG(dim::Int,l::Int,m::Int,b::Int=_maxlevel[dim-1]) = OctantBWG(dim,Int32(l),Int32(m),Int32(b))
OctantBWG(dim::Int,l::Int,m::Int,b::Int32) = OctantBWG(dim,Int32(l),Int32(m),b)
OctantBWG(dim::Int,l::Int32,m::Int,b::Int32) = OctantBWG(dim,l,Int32(m),b)
OctantBWG(level::Int,coords::NTuple) = OctantBWG(Int32(level),Int32.(coords))
OctantBWG(level::Int32,coords::NTuple) = OctantBWG(level,Int32.(coords))
function OctantBWG(level::Int32, coords::NTuple{dim,Int32}) where dim
    dim == 2 ? OctantBWG{2,4,4,Int32}(level,coords) : OctantBWG{3,8,6,Int32}(level,coords)
end

# From BWG 2011
# > The octant coordinates are stored as integers of a fixed number b of bits,
# > where the highest (leftmost) bit represents the first vertical level of the
# > octree (counting the root as level zero), the second highest bit the second level of the octree, and so on.
# Morton Index can thus be constructed by interleaving the integer bits:
# m(Oct) := (y_b,x_b,y_b-1,x_b-1,...y0,x0)_2
# further we assume the following
# > Due to the two-complement representation of integers in practically all current hardware,
# > where the highest digit denotes the negated appropriate power of two, bitwise operations as used,
# > for example, in Algorithm 1 yield the correct result even for negative coordinates.
# also from BWG 2011
# TODO: use LUT method from https://www.forceflow.be/2013/10/07/morton-encodingdecoding-through-bit-interleaving-implementations/
function morton(octant::OctantBWG{dim,N,M,T},l::T,b::T) where {dim,N,M,T<:Integer}
    o = one(T)
    z = zero(T)
    id = zero(widen(eltype(octant.xyz)))
    loop_length = (sizeof(typeof(id))*T(8)) ÷ dim - o
    for i in z:loop_length
        for d in z:dim-o
            # first shift extract i-th bit and second shift inserts it at interleaved index 
            id = id | ((octant.xyz[d+o] & (o << i)) << ((dim-o)*i+d))
        end
    end
    # discard the bit information about deeper levels
    return (id >> ((b-l)*dim))+o
end
morton(octant::OctantBWG{dim,N,M,T1},l::T2,b::T3) where {dim,N,M,T1<:Integer,T2<:Integer,T3<:Integer} = morton(octant,T1(l),T1(b))

Base.zero(::Type{OctantBWG{3, 8, 6}}) = OctantBWG(3, 0, 1)
Base.zero(::Type{OctantBWG{2, 4, 4}}) = OctantBWG(2, 0, 1)

ncorners(::Type{OctantBWG{dim,N,M,T}}) where {dim,N,M,T} = N
ncorners(o::OctantBWG) = ncorners(typeof(o))
nchilds(::Type{OctantBWG{dim,N}}) where {dim,N} = N
nchilds(o::OctantBWG) = nchilds(typeof(o))# Follow z order, x before y before z for faces, edges and corners

# TODO: consider list instead of Vector datastructure
struct OctreeBWG{dim,N,M,T} <: AbstractAdaptiveCell{dim,N,M}
    leaves::Vector{OctantBWG{dim,N,M,T}}
    #maximum refinement level
    b::T
    nodes::NTuple{N,Int}
end

function refine!(octree::OctreeBWG{dim,N,M,T}, pivot_octant::OctantBWG{dim,N,M,T}) where {dim,N,M,T<:Integer}
    o = one(T)
    # really the only way?
    leave_idx = findfirst(x->x==pivot_octant,octree.leaves)
    # how to obtain id from morton index ?
    old_octant = popat!(octree.leaves,leave_idx)
    start_child_id = morton(old_octant,old_octant.l+o,octree.b)
    end_child_id = start_child_id + N-o
    for child_mort_id in start_child_id:end_child_id
        insert!(octree.leaves,leave_idx,OctantBWG(dim,old_octant.l+o,child_mort_id,octree.b))
        leave_idx += 1
    end
end

function coarsen!(octree::OctreeBWG{dim,N,M,T}, o::OctantBWG{dim,N,M,T}) where {dim,N,M,T<:Integer}
    _two = T(2)
    leave_idx = findfirst(x->x==o,octree.leaves)
    shift = child_id(o,octree.b) - one(T)
    if shift != zero(T)
        old_morton = morton(o,o.l,octree.b)
        o = OctantBWG(dim,o.l,old_morton,octree.b)
    end
    window_start = leave_idx - shift
    window_length = _two^dim - one(T)
    new_octant = parent(o, octree.b)
    octree.leaves[leave_idx - shift] = new_octant
    deleteat!(octree.leaves,leave_idx-shift+one(T):leave_idx-shift+window_length)
end

OctreeBWG{3,8,6}(nodes::NTuple,b=_maxlevel[2]) = OctreeBWG{3,8,6,Int32}([zero(OctantBWG{3,8,6})],Int32(b),nodes)
OctreeBWG{2,4,4}(nodes::NTuple,b=_maxlevel[1]) = OctreeBWG{2,4,4,Int32}([zero(OctantBWG{2,4,4})],Int32(b),nodes)
OctreeBWG(cell::Quadrilateral,b=_maxlevel[2]) = OctreeBWG{2,4,4}(cell.nodes,b)
OctreeBWG(cell::Hexahedron,b=_maxlevel[1]) = OctreeBWG{3,8,6}(cell.nodes,b)

Base.length(tree::OctreeBWG) = length(tree.leaves)

function inside(tree::OctreeBWG{dim},oct::OctantBWG{dim}) where dim
    maxsize = _maximum_size(tree.b)
    outside = any(xyz -> xyz >= maxsize, oct.xyz) || any(xyz -> xyz < 0, oct.xyz)
    return !outside
end

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

function ForestBWG(grid::AbstractGrid{dim},b=_maxlevel[dim-1]) where dim
    cells = getcells(grid)
    C = eltype(cells)
    @assert isconcretetype(C)
    @assert (C == Quadrilateral && dim == 2) || (C == Hexahedron && dim == 3)
    topology = ExclusiveTopology(cells)
    cells = OctreeBWG.(grid.cells,b)
    nodes = getnodes(grid)
    cellsets = getcellsets(grid)
    nodesets = getnodesets(grid)
    facesets = getfacesets(grid)
    edgesets = getedgesets(grid)
    vertexsets = getvertexsets(grid)
    return ForestBWG(cells,nodes,cellsets,nodesets,facesets,edgesets,vertexsets,topology)
end

getneighborhood(forest::ForestBWG,idx) = getneighborhood(forest.topology,forest,idx)

function getncells(grid::ForestBWG)
    numcells = 0
    for tree in grid.cells
        numcells += length(tree)
    end
    return numcells
end

function getcells(forest::ForestBWG{dim}) where dim
    celltype = dim == 2 ? OctantBWG{2,4,4,Int32} : OctantBWG{3,8,6,Int32}
    ncells = getncells(forest)
    cellvector = Vector{celltype}(undef,ncells)
    o = one(Int32)
    cellid = o
    for tree in forest.cells
        for leaf in tree.leaves
            cellvector[cellid] = leaf
            cellid += o
        end
    end
    return cellvector
end

function getcells(forest::ForestBWG{dim}, cellid::Int)  where dim
    @warn "Slow dispatch, consider to call `getcells(forest)` once instead" maxlog=1 #TODO doc page for performance
    #TODO should nleaves be saved by forest?
    nleaves = length.(forest.cells) # cells=trees
    nleaves_cumsum = cumsum(nleaves)
    k = findfirst(x->cellid<=x,nleaves_cumsum)
    #TODO is this actually correct?
    leafid = k == 1 ? cellid : cellid - (nleaves_cumsum[k] - nleaves[k])
    return forest.cells[k].leaves[leafid]
end

getcelltype(grid::ForestBWG) = eltype(grid.cells)
getcelltype(grid::ForestBWG, i::Int) = eltype(grid.cells) # assume for now same cell type TODO

function getnodes(forest::ForestBWG{dim,C,T}) where {dim,C,T}
    nodes = Vector{Node{dim,Int32}}()
    sizehint!(nodes,getncells(forest)*2^dim) # TODO worth it? max is everything duplicate
    for (k,tree) in enumerate(forest.cells)
        for leaf in tree.leaves
            for c in 1:ncorners(leaf)
                neighbor = corner_neighbor(leaf,c,tree.b)
                if inside(tree,neighbor) # checks if neighbor is in boundary of tree (w.r.t. octree coordinates)
                    # I think the below is valid
                    neighbor_morton = morton(neighbor,neighbor.l,tree.b)
                    leaf_morton = morton(leaf,leaf.l,tree.b)
                    # If the participating neighbor has higher morton id assign new node
                    if leaf_morton < neighbor_morton
                        push!(nodes, Node(transform_corner(forest,k,c,leaf).xyz))
                    end
                else
                    #TODO I don't know how to handle the other case, the below doesn't work
                    lowest_octree = true
                    for f in corner_face_participation(dim,c)
                        k′ = getneighborhood(forest,FaceIndex(k,f))
                        if isempty(k′)
                            continue
                        else
                            k′ = k′[1][1] # always half face, ugly TODO
                        end
                        if k′ < k
                            lowest_octree = false
                            break
                        end
                    end
                    if lowest_octree
                        #transform needs the neighbor c
                       push!(nodes,Node(transform_corner(forest,k,c,leaf).xyz))
                    end
                end
            end
        end
    end
    return nodes
end

function Base.show(io::IO, ::MIME"text/plain", agrid::ForestBWG)
    println(io, "ForestBWG with ")
    println(io, "   $(getncells(agrid)) cells")
    println(io, "   $(length(agrid.cells)) trees")
end

"""
    child_id(octant::OctantBWG, b::Integer)
Given some OctantBWG `octant` and maximum refinement level `b`, compute the child_id of `octant`
note the following quote from Bursedde et al:
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

function parent(octant::OctantBWG{dim,N,M,T}, b::Integer=_maxlevel[dim-1]) where {dim,N,M,T}
    if octant.l > zero(T)
        h = T(_compute_size(b,octant.l))
        l = octant.l - one(T)
        return OctantBWG(l,octant.xyz .& ~h)
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
function descendants(octant::OctantBWG{dim,N,M,T}, b::Integer=_maxlevel[dim-1]) where {dim,N,M,T}
    l1 = b; l2 = b
    h = T(_compute_size(b,octant.l))
    return OctantBWG(l1,octant.xyz), OctantBWG(l2,octant.xyz .+ (h-one(T)))
end

function face_neighbor(octant::OctantBWG{dim,N,M,T}, f::T, b::T=_maxlevel[dim-1]) where {dim,N,M,T<:Integer}
    l = octant.l
    h = T(_compute_size(b,octant.l))
    x,y,z = octant.xyz
    x += ((f == T(1)) ? -h : ((f == T(2)) ? h : zero(T)))
    y += ((f == T(3)) ? -h : ((f == T(4)) ? h : zero(T)))
    z += ((f == T(5)) ? -h : ((f == T(6)) ? h : zero(T)))
    dim == 2 ? OctantBWG(l,(x,y)) : OctantBWG(l,(x,y,z))
end
face_neighbor(o::OctantBWG{dim,N,M,T1}, f::T2, b::T3) where {dim,N,M,T1<:Integer,T2<:Integer,T3<:Integer} = face_neighbor(o,T1(f),T1(b))

#TODO: this is not working in 2d as of now, indices used in the paper confuse me
function transform_face(forest::ForestBWG, k::T1, f::T1, o::OctantBWG{dim,N,M,T2}) where {dim,N,M,T1<:Integer,T2<:Integer}
    _one = one(T2)
    _two = T2(2)
    #currently rotation not encoded
    kprime, fprime = getneighborhood(forest,FaceIndex(k,f))[1]
    a₂ = f ÷ 2; b₂ = fprime ÷ 2
    sprime = _one - ((f & _one) ⊻ (fprime & _one))
    s = zeros(T2,2)
    b = zeros(T2,3)
    r = 0 #no rotation information in face_neighbor currently
    if dim == 2
        a₀ = 1 - a₂; b₀ = 1 - b₂; s[1] = r #no rotation as of now
    else
        a₀ = (f < 3) ? 1 : 0; a₁ = (f < 5) ? 2 : 1
        #u = ℛ[1,f] ⊻ ℛ[1,fprime] ⊻ T2((r == 1) | (r == 3))
        b[1] = (fprime < 3) ? 1 : 0; b[2] = (fprime < 5) ? 2 : 1
        #v = T2(ℛ[f,fprime] == 1)
        s[1] = r & 2; s[2] = r & 3
    end
    b = forest.cells[1].b
    l = o.l; g = 2^b - 2^(b-l)
    x = T2((s[1] == 1) ? o.xyz[1] : g - o.xyz[1])
    y = T2((s[2] == 1) ? o.xyz[2] : g - o.xyz[2])
    z = T2((_two*(fprime & 1) - 1)*2^b + sprime*g + (1-2*sprime)*o.xyz[2])
    if dim == 2
        return OctantBWG(l,(x,z))
    else
        return OctantBWG(l,(x,y,z))
    end
end

"""
    transform_corner(forest,k,c',oct)
    transform_corner(forest,v::VertexIndex,oct)

Algorithm 12 in p4est paper to transform corner into different octree coordinate system
Note: in Algorithm 12 is c as a argument, but it's never used, therefore I removed it
"""
function transform_corner(forest::ForestBWG,k::T1,c′::T1,oct::OctantBWG{dim,N,M,T2}) where {dim,N,M,T1<:Integer,T2<:Integer}
    # make a dispatch that returns only the coordinates?
    b = forest.cells[k].b
    l = oct.l; g = 2^b - 2^(b-l)
    _inside = inside(forest.cells[k],oct)
    h⁻ = _inside ? 0 : -2^(b-l); h⁺ = _inside ? g : 2^b
    xyz = ntuple(i->((c′-1) & 2^(i-1) == 0) ? h⁻ : h⁺,dim)
    return OctantBWG(l,xyz)
end

transform_corner(forest::ForestBWG,v::VertexIndex,oct::OctantBWG) = transform_corner(forest,v[1],v[2],oct)

"""
    edge_neighbor(octant::OctantBWG, e::Integer, b::Integer)
Computes the edge neighbor octant which is only connected by the edge `e` to `octant`
"""
function edge_neighbor(octant::OctantBWG{3,N,M,T}, e::T, b::T=_maxlevel[2]) where {N,M,T<:Integer}
    @assert 1 ≤ e ≤ 12
    e -= one(T)
    l = octant.l
    _one = one(T)
    _two = T(2)
    h = T(_compute_size(b,octant.l))
    ox,oy,oz = octant.xyz
    case = e ÷ T(4)
    if case == zero(T)
        x = ox
        y = oy + (_two*(e & _one) - one(T))*h
        z = oz + ((e & _two) - _one)*h
        return OctantBWG(l,(x,y,z))
    elseif case == one(T)
        x = ox  + (_two*(e & _one) - _one)*h
        y = oy
        z = oz + ((e & _two) - _one)*h
        return OctantBWG(l,(x,y,z))
    elseif case == _two
        x = ox + (_two*(e & _one) - _one)*h
        y = oy + ((e & _two) - _one)*h
        z = oz
        return OctantBWG(l,(x,y,z))
    else
        error("edge case not found")
    end
end
edge_neighbor(o::OctantBWG{3,N,M,T1}, e::T2, b::T3) where {N,M,T1<:Integer,T2<:Integer,T3<:Integer} = edge_neighbor(o,T1(e),T1(b))

"""
    corner_neighbor(octant::OctantBWG, c::Integer, b::Integer)
Computes the corner neighbor octant which is only connected by the corner `c` to `octant`
"""
function corner_neighbor(octant::OctantBWG{3,N,M,T}, c::T, b::T=_maxlevel[2]) where {N,M,T<:Integer}
    c -= one(T)
    l = octant.l
    h = T(_compute_size(b,octant.l))
    ox,oy,oz = octant.xyz
    _one = one(T)
    _two = T(2)
    x = ox + (_two*(c & _one) - _one)*h
    y = oy + ((c & _two) - _one)*h
    z = oz + ((c & T(4))÷_two - _one)*h
    return OctantBWG(l,(x,y,z))
end

function corner_neighbor(octant::OctantBWG{2,N,M,T}, c::T, b::T=_maxlevel[1]) where {N,M,T<:Integer}
    c -= one(T)
    l = octant.l
    h = _compute_size(b,octant.l)
    ox,oy = octant.xyz
    _one = one(T)
    _two = T(2)
    x = ox + (_two*(c & _one) - _one)*h
    y = oy + ((c & _two) - _one)*h
    return OctantBWG(l,(x,y))
end
corner_neighbor(o::OctantBWG{dim,N,M,T1}, c::T2, b::T3) where {dim,N,M,T1<:Integer,T2<:Integer,T3<:Integer} = corner_neighbor(o,T1(c),T1(b))

function corner_face_participation(dim::T,c::T) where T<:Integer
    if dim == 2
        return 𝒱₂_perm[findall(x->c ∈ x, eachrow(𝒱₂))]
    else
        return 𝒱₃_perm[findall(x->c ∈ x, eachrow(𝒱₃))]
    end
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
_maximum_size(b::Integer) = 2^(b)
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

const 𝒱₂_perm = [4
                 2
                 1
                 3]

const 𝒱₃_perm = [2
                 4
                 3
                 5
                 1
                 6]

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
