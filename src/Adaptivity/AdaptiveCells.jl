abstract type AbstractAdaptiveGrid{dim} <: AbstractGrid{dim} end
abstract type AbstractAdaptiveCell{refshape <: AbstractRefShape} <: AbstractCell{refshape} end

_maxlevel = [30,19]

function set_maxlevel(dim::Integer,maxlevel::Integer)
    _maxlevel[dim-1] = maxlevel
end

struct OctantBWG{dim, N, M, T} <: AbstractCell{RefHypercube{dim}}
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
root(dim::T) where T<:Integer = zero(OctantBWG{dim,dim^2,2*dim})

ncorners(::Type{OctantBWG{dim,N,M,T}}) where {dim,N,M,T} = N
ncorners(o::OctantBWG) = ncorners(typeof(o))
nchilds(::Type{OctantBWG{dim,N,M,T}}) where {dim,N,M,T} = N
nchilds(o::OctantBWG) = nchilds(typeof(o))# Follow z order, x before y before z for faces, edges and corners

Base.isequal(o1::OctantBWG, o2::OctantBWG) = (o1.l == o2.l) && (o1.xyz == o2.xyz)
"""
    o1::OctantBWG < o2::OctantBWG
Implements Algorithm 2.1 of IBWG 2015.
Checks first if mortonid is smaller and later if level is smaller.
Thus, ancestors precede descendants (preordering).
"""
function Base.isless(o1::OctantBWG, o2::OctantBWG)
    if o1.xyz != o2.xyz
        #TODO verify b=o1.l/b=o2.l as argument potential bug otherwise
        return morton(o1,o1.l,o1.l) < morton(o2,o2.l,o2.l)
    else
        return o1.l < o2.l
    end
end

function children(octant::OctantBWG{dim,N,M,T}, b::Integer) where {dim,N,M,T}
    o = one(T)
    _nchilds = nchilds(octant)
    startid = morton(octant,octant.l+o,b)
    endid = startid + _nchilds + o
    return ntuple(i->OctantBWG(dim,octant.l+o,(startid:endid)[i],b),_nchilds)
end

abstract type OctantIndex{T<:Integer} end
Base.isequal(i1::T,i2::T) where T<:OctantIndex = i1.idx == i2.idx #same type
Base.isequal(i1::T1,i2::T2) where {T1<:OctantIndex,T2<:OctantIndex} = false #different type

struct OctantCornerIndex{T} <: OctantIndex{T}
    idx::T
end
Base.hash(idx::OctantCornerIndex) = Base.hash((0,idx.idx))
Base.show(io::IO, ::MIME"text/plain", c::OctantCornerIndex) = print(io, "O-Corner $(c.idx)")
Base.show(io::IO, c::OctantCornerIndex) = print(io, "O-Corner $(c.idx)")

struct OctantEdgeIndex{T} <: OctantIndex{T}
    idx::T
end
Base.hash(idx::OctantEdgeIndex) = Base.hash((1,idx.idx))
Base.show(io::IO, ::MIME"text/plain", e::OctantEdgeIndex) = print(io, "O-Edge $(e.idx)")
Base.show(io::IO, e::OctantEdgeIndex) = print(io, "O-Edge $(e.idx)")

struct OctantFaceIndex{T} <: OctantIndex{T}
    idx::T
end
Base.hash(idx::OctantFaceIndex) = Base.hash((2,idx.idx))
Base.show(io::IO, ::MIME"text/plain", f::OctantFaceIndex) = print(io, "O-Face $(f.idx)")
Base.show(io::IO, f::OctantFaceIndex) = print(io, "O-Face $(f.idx)")

vertex(octant::OctantBWG, c::OctantCornerIndex, b::Integer) = vertex(octant,c.idx,b)
function vertex(octant::OctantBWG{dim,N,M,T}, c::Integer, b::Integer) where {dim,N,M,T}
    h = T(_compute_size(b,octant.l))
    return ntuple(d->((c-1) & (2^(d-1))) == 0 ? octant.xyz[d] : octant.xyz[d] + h ,dim)
end

function vertices(octant::OctantBWG{dim},b::Integer) where {dim}
    _nvertices = 2^dim
    return ntuple(i->vertex(octant,i,b),_nvertices)
end

vertex(octant::OctantBWG, f::OctantFaceIndex, b::Integer) = vertex(octant,f.idx,b)
function face(octant::OctantBWG{2}, f::Integer, b::Integer)
    cornerid = view(𝒱₂,f,:)
    return ntuple(i->vertex(octant, cornerid[i], b),2)
end

function face(octant::OctantBWG{3}, f::Integer, b::Integer)
    cornerid = view(𝒱₃,f,:)
    return ntuple(i->vertex(octant, cornerid[i], b),4)
end

function faces(octant::OctantBWG{dim}, b::Integer) where dim
    _nfaces = 2*dim
    return ntuple(i->face(octant,i,b),_nfaces)
end

vertex(octant::OctantBWG, e::OctantEdgeIndex, b::Integer) = vertex(octant,e.idx,b)
function edge(octant::OctantBWG{3}, e::Integer, b::Integer)
    cornerid = view(𝒰,e,:)
    return ntuple(i->vertex(octant,cornerid[i], b),2)
end

"""
    boundaryset(o::OctantBWG{2}, i::Integer, b::Integer
implements two dimensional boundaryset table from Fig.4.1 IBWG 2015
TODO: could be done little bit less ugly
"""
function boundaryset(o::OctantBWG{2,N,M,T}, i::Integer, b::Integer) where {N,M,T}
    if i==1
        return Set((OctantCornerIndex(1),OctantFaceIndex(1),OctantFaceIndex(3)))
    elseif i==2
        return Set((OctantCornerIndex(2),OctantFaceIndex(2),OctantFaceIndex(3)))
    elseif i==3
        return Set((OctantCornerIndex(3),OctantFaceIndex(1),OctantFaceIndex(4)))
    elseif i==4
        return Set((OctantCornerIndex(4),OctantFaceIndex(2),OctantFaceIndex(4)))
    else
        throw("no boundary")
    end
end

"""
    boundaryset(o::OctantBWG{3}, i::Integer, b::Integer
implements three dimensional boundaryset table from Fig.4.1 IBWG 2015
TODO: could be done little bit less ugly
"""
function boundaryset(o::OctantBWG{3,N,M,T}, i::Integer, b::Integer) where {N,M,T}
    if i==1
        return Set((OctantCornerIndex(1),OctantEdgeIndex(1),OctantEdgeIndex(5),OctantEdgeIndex(9), OctantFaceIndex(1),OctantFaceIndex(3),OctantFaceIndex(5)))
    elseif i==2
        return Set((OctantCornerIndex(2),OctantEdgeIndex(1),OctantEdgeIndex(6),OctantEdgeIndex(10),OctantFaceIndex(2),OctantFaceIndex(3),OctantFaceIndex(5)))
    elseif i==3
        return Set((OctantCornerIndex(3),OctantEdgeIndex(2),OctantEdgeIndex(5),OctantEdgeIndex(11),OctantFaceIndex(1),OctantFaceIndex(4),OctantFaceIndex(5)))
    elseif i==4
        return Set((OctantCornerIndex(4),OctantEdgeIndex(2),OctantEdgeIndex(6),OctantEdgeIndex(12),OctantFaceIndex(2),OctantFaceIndex(4),OctantFaceIndex(5)))
    elseif i==5
        return Set((OctantCornerIndex(5),OctantEdgeIndex(3),OctantEdgeIndex(7),OctantEdgeIndex(9), OctantFaceIndex(1),OctantFaceIndex(3),OctantFaceIndex(6)))
    elseif i==6
        return Set((OctantCornerIndex(6),OctantEdgeIndex(3),OctantEdgeIndex(8),OctantEdgeIndex(10),OctantFaceIndex(2),OctantFaceIndex(3),OctantFaceIndex(6)))
    elseif i==7
        return Set((OctantCornerIndex(7),OctantEdgeIndex(4),OctantEdgeIndex(7),OctantEdgeIndex(11),OctantFaceIndex(1),OctantFaceIndex(4),OctantFaceIndex(6)))
    elseif i==8
        return Set((OctantCornerIndex(8),OctantEdgeIndex(4),OctantEdgeIndex(8),OctantEdgeIndex(12),OctantFaceIndex(2),OctantFaceIndex(4),OctantFaceIndex(6)))
    else
        throw("no boundary")
    end
end

"""
    find_range_boundaries(f::OctantBWG{dim,N,M,T}, l::OctantBWG{dim,N,M,T}, s::OctantBWG{dim,N,M,T}, idxset, b)
    find_range_boundaries(s::OctantBWG{dim,N,M,T}, idxset, b)
Algorithm 4.2 of IBWG 2015
TODO: write tests
"""
function find_range_boundaries(f::OctantBWG{dim,N,M,T1}, l::OctantBWG{dim,N,M,T1}, s::OctantBWG{dim,N,M,T1}, idxset::Set{OctantIndex{T2}}, b) where {dim,N,M,T1,T2}
    o = one(T1)
    if isempty(idxset) || s.l == b
        return idxset
    end
    j = ancestor_id(f,s.l+o,b); k = ancestor_id(l,s.l+o,b)
    boundary_j = boundaryset(s,j,b)
    kidz = children(s,b)
    if j==k
        return find_range_boundaries(f,l,kidz[j],idxset ∩ boundary_j,b)
    end
    idxset_match = Set{OctantIndex{T2}}()
    for i in (j+o):(k-o)
        union!(idxset_match,idxset ∩ boundaryset(s,i,b))
    end
    boundary_k = boundaryset(s,k,b)
    idxset_match_j = setdiff((idxset ∩ boundary_j),idxset_match)
    fj, lj = descendants(kidz[j],b)
    if fj != f
        idxset_match_j = find_range_boundaries(f,lj,kidz[j],idxset_match_j,b)
    end
    idxset_match_k = setdiff(setdiff((idxset ∩ boundary_k),idxset_match),idxset_match_j)
    fk, lk = descendants(kidz[k],b)
    if lk != l
        idxset_match_k = find_range_boundaries(fk,l,kidz[k],idxset_match_k,b)
    end
    return idxset_match ∪ idxset_match_j ∪ idxset_match_k
end

#for convenience, should probably changed to parent(s) until parent(s)==root and then descendants(root)
function find_range_boundaries(s::OctantBWG, idxset, b)
    f,l = descendants(s,b)
    return find_range_boundaries(f,l,s,idxset,b)
end

function isrelevant(xyz::NTuple{dim,T},leafsuppₚ::Set{<:OctantBWG}) where {dim,T}
    ###### only relevant for distributed
    #for all s in leafsuppₚ
    #    if s in 𝒪ₚ
    #        return true
    #    else
    #        check stuff Algorithm 5.1 line 4-5
    #    end
    #end
    return true
end

struct OctreeBWG{dim,N,M,T} <: AbstractAdaptiveCell{RefHypercube{dim}}
    leaves::Vector{OctantBWG{dim,N,M,T}}
    #maximum refinement level
    b::T
    nodes::NTuple{N,Int}
end

function refine!(octree::OctreeBWG{dim,N,M,T}, pivot_octant::OctantBWG{dim,N,M,T}) where {dim,N,M,T<:Integer}
    @assert pivot_octant.l + 1 <= octree.b
    o = one(T)
    # TODO replace this with recursive search function
    leave_idx = findfirst(x->x==pivot_octant,octree.leaves)
    old_octant = popat!(octree.leaves,leave_idx)
    _children = children(pivot_octant,octree.b)
    for child in _children
        insert!(octree.leaves,leave_idx,child)
        leave_idx += 1
    end
end

function refine_all(forest::ForestBWG,l)
   for tree in forest.cells
      for leaf in tree.leaves
          if leaf.l != l-1 #maxlevel
              continue
          else
              refine!(tree,leaf)
          end
      end
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
    split_array(octree::OctreeBWG, a::OctantBWG)
    split_array(octantarray, a::OctantBWG, b::Integer)
Algorithm 3.3 of IBWG2015. Efficient binary search
"""
function split_array(octantarray, a::OctantBWG{dim,N,M,T}, b::Integer) where {dim,N,M,T}
    o = one(T)
    𝐤 = T[i==1 ? 1 : length(octantarray)+1 for i in 1:2^dim+1]
    for i in 2:2^dim
        m = 𝐤[i-1]
        while m < 𝐤[i]
            n = m + (𝐤[i] - m)÷2
            c = ancestor_id(octantarray[n], a.l+o, b)
            if c < i
                m = n+1
            else
                for j in i:c
                    𝐤[j] = n
                end
            end
        end
    end
    #TODO non-allocating way?
    return ntuple(i->view(octantarray,𝐤[i]:𝐤[i+1]-1),2^dim)
end

split_array(tree::OctreeBWG, a::OctantBWG) = split_array(tree.leaves, a, tree.b)

function search(octantarray, a::OctantBWG{dim,N,M,T1}, idxset::Vector{T2}, b::Integer, Match=match) where {dim,N,M,T1<:Integer,T2}
    isempty(octantarray) && return
    isleaf = (length(octantarray) == 1 && a ∈ octantarray) ? true : false
    idxset_match = eltype(idxset)[]
    for q in idxset
        if Match(a,isleaf,q,b)
            push!(idxset_match,q)
        end
    end
    if isempty(idxset_match) && !isleaf
        𝐇 = split_array(octantarray,a,b)
        _children = children(a,b)
        for (child,h) in zip(_children,𝐇)
            search(h,child,idxset_match,b)
        end
    end
    return idxset_match
end

search(tree::OctreeBWG, a::OctantBWG, idxset, Match=match) = search(tree.leaves, a, idxset, tree.b, match)

"""
    match(o::OctantBWG, isleaf::Bool, q)
from IBWG2015
> match returns true if there is a leaf r ∈ 𝒪 that is a descendant of o
> such that match_q(r) = true, and is allowed to return a false positive
> (i.e., true even if match_q(r) = false for all descendants leaves of o)
> if isleaf=true, then the return  value of match is irrelevant
I don't understand what of a to check against index q
"""
function match(o::OctantBWG, isleaf::Bool, q, b)
    isleaf && (return true)
    println(q)
    println(o)
    return false
end

"""
    ForestBWG{dim, C<:AbstractAdaptiveCell, T<:Real} <: AbstractAdaptiveGrid{dim}
`p4est` adaptive grid implementation based on Burstedde, Wilcox, Ghattas [2011]
and Isaac, Burstedde, Wilcox, Ghattas [2015]
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
    #TODO remove that later by for loop or IBWG 2015 iterator approach
    nleaves_cumsum = cumsum(nleaves)
    k = findfirst(x->cellid<=x,nleaves_cumsum)
    #TODO is this actually correct?
    leafid = k == 1 ? cellid : cellid - (nleaves_cumsum[k] - nleaves[k])
    return forest.cells[k].leaves[leafid]
end

getcelltype(grid::ForestBWG) = eltype(grid.cells)
getcelltype(grid::ForestBWG, i::Int) = eltype(grid.cells) # assume for now same cell type TODO
function transform_pointBWG(forest::ForestBWG{dim}, k::Integer, vertex::NTuple{dim,T}) where {dim,T}
    tree = forest.cells[k]
    cellnodes = getnodes(forest,collect(tree.nodes)) .|> get_node_coordinate
    vertex = vertex .* (2/(2^tree.b)) .- 1
    octant_physical_coordinates = sum(j-> cellnodes[j] * Ferrite.shape_value(Lagrange{Ferrite.RefHypercube{dim},1}(),Vec{dim}(vertex),j),1:length(cellnodes)) 
    return Vec{dim}(octant_physical_coordinates)
end

transform_pointBWG(forest, vertices) = transform_pointBWG.((forest,), first.(vertices), last.(vertices))

#TODO: this function should wrap the LNodes Iterator of IBWG2015
#TODO: need 𝒱₃ perm tables
function getnodes(forest::ForestBWG{dim,C,T}) where {dim,C,T}
    nodes = Set{Tuple{Int,NTuple{dim,Int32}}}()
    for (k,tree) in enumerate(forest.cells)
        for leaf in tree.leaves
            _vertices = vertices(leaf,tree.b)
            for v in _vertices
                push!(nodes,(k,v))
            end
        end
    end
    for (k,tree) in enumerate(forest.cells)
        _vertices = vertices(root(dim),tree.b)
        for (vi,v) in enumerate(_vertices)
            vertex_neighbor =  forest.topology.vertex_neighbor[k,vi]
            if length(vertex_neighbor) == 0
                continue
            end
            if k > vertex_neighbor[1][1]
                delete!(nodes,(k,v))
            end
        end
        _faces = faces(root(dim),tree.b)
        for (fi,f) in enumerate(_faces) # fi in p4est notation
            face_neighbor =  forest.topology.face_neighbor[k,𝒱₂_perm[fi]]
            if length(face_neighbor) == 0
                continue
            end
            k′ = face_neighbor[1][1]
            if k > k′
                for leaf in tree.leaves
                    for v in vertices(leaf,tree.b)
                        if fi < 3
                            if v[1] == f[1][1] == f[2][1]
                                cache_octant = OctantBWG(leaf.l,v)
                                cache_octant = transform_face(forest,k′,𝒱₂_perm_inv[face_neighbor[1][2]],cache_octant) # after transform
                                if (k′,cache_octant.xyz) ∈ nodes
                                    delete!(nodes,(k,v))
                                end
                            end
                        elseif fi < 5
                            if v[2] == f[1][2] == f[2][2]
                                cache_octant = OctantBWG(leaf.l,v)
                                cache_octant = transform_face(forest,k′,𝒱₂_perm_inv[face_neighbor[1][2]],cache_octant) # after transform
                                if (k′,cache_octant.xyz) ∈ nodes
                                    delete!(nodes,(k,v))
                                end
                            end
                        else
                            @error "help"
                            if v[3] == f[1][3] == f[2][3]
                                cache_octant = OctantBWG(leaf.l,v)
                                cache_octant = transform_face(forest,k′,𝒱₂_perm_inv[face_neighbor[1][2]],cache_octant) # after transform
                                if (k′,cache_octant.xyz) ∈ nodes
                                    delete!(nodes,(k,v))
                                end
                            end
                        end
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
function child_id(octant::OctantBWG{dim,N,M,T},b::Integer=_maxlevel[2]) where {dim,N,M,T<:Integer}
    i = 0x00
    t = T(2)
    z = zero(T)
    h = T(_compute_size(b,octant.l))
    xyz = octant.xyz
    for j in 0:(dim-1)
        i = i | ((xyz[j+1] & h) != z ? t^j : z)
    end
    return i+0x01
end

"""
    ancestor_id(octant::OctantBWG, l::Integer, b::Integer)
Algorithm 3.2 of IBWG 2015 that generalizes `child_id` for different queried levels.
Applied to a single octree, i.e. the array of leaves, yields a monotonic sequence
"""
function ancestor_id(octant::OctantBWG{dim,N,M,T}, l::Integer, b::Integer=_maxlevel[dim-1]) where {dim,N,M,T<:Integer}
    @assert 0 < l ≤ octant.l
    i = 0x00
    t = T(2)
    z = zero(T)
    h = T(_compute_size(b,l))
    for j in 0:(dim-1)
       i = i | ((octant.xyz[j+1] & h) != z ? t^j : z)
    end
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

function face_neighbor(octant::OctantBWG{3,N,M,T}, f::T, b::T=_maxlevel[2]) where {N,M,T<:Integer}
    l = octant.l
    h = T(_compute_size(b,octant.l))
    x,y,z = octant.xyz
    x += ((f == T(1)) ? -h : ((f == T(2)) ? h : zero(T)))
    y += ((f == T(3)) ? -h : ((f == T(4)) ? h : zero(T)))
    z += ((f == T(5)) ? -h : ((f == T(6)) ? h : zero(T)))
    return OctantBWG(l,(x,y,z))
end
function face_neighbor(octant::OctantBWG{2,N,M,T}, f::T, b::T=_maxlevel[1]) where {N,M,T<:Integer}
    l = octant.l
    h = T(_compute_size(b,octant.l))
    x,y = octant.xyz
    x += ((f == T(1)) ? -h : ((f == T(2)) ? h : zero(T)))
    y += ((f == T(3)) ? -h : ((f == T(4)) ? h : zero(T)))
    return OctantBWG(l,(x,y))
end
face_neighbor(o::OctantBWG{dim,N,M,T1}, f::T2, b::T3) where {dim,N,M,T1<:Integer,T2<:Integer,T3<:Integer} = face_neighbor(o,T1(f),T1(b))

#TODO: this is working for 2D except rotation, 3D I don't know
function transform_face(forest::ForestBWG, k::T1, f::T1, o::OctantBWG{dim,N,M,T2}) where {dim,N,M,T1<:Integer,T2<:Integer}
    _one = one(T2)
    _two = T2(2)
    #currently rotation not encoded
    perm = (dim == 2 ? 𝒱₂_perm : 𝒱₃_perm)
    kprime, fprime = getneighborhood(forest,FaceIndex(k,perm[f]))[1]
    fprime = 𝒱₂_perm_inv[fprime]
    sprime = _one - (((f - _one) & _one) ⊻ ((fprime - _one) & _one))
    s = zeros(T2,3)
    b = zeros(T2,3)
    a = zeros(T2,3)
    r = zero(T2)  #no rotation information in face_neighbor currently
    a[3] = (f-_one) ÷ 2; b[3] = (fprime-_one) ÷ 2
    if dim == 2
        a[1] = 1 - a[3]; b[1] = 1 - b[3]; s[1] = r #no rotation as of now
    else
        a[1] = (f < 3) ? 1 : 0; a[2] = (f < 5) ? 2 : 1
        #u = ℛ[1,f] ⊻ ℛ[1,fprime] ⊻ T2((r == 1) | (r == 3))
        b[1] = (fprime < 3) ? 1 : 0; b[2] = (fprime < 5) ? 2 : 1 # r = 0 -> index 1
        #v = T2(ℛ[f,fprime] == 1)
        s[1] = r & 1; s[2] = r & 2
    end
    maxlevel = forest.cells[1].b
    l = o.l; g = 2^maxlevel - 2^(maxlevel-l)
    xyz = zeros(T2,dim)
    xyz[b[1] + 1] = T2((s[1] == 0) ? o.xyz[a[1] + 1] : g - o.xyz[a[1] + 1])
    xyz[b[3] + 1] = T2((_two*(fprime & 1) - 1)*2^maxlevel + sprime*g + (1-2*sprime)*o.xyz[a[3] + 1])
    if dim == 2
        return OctantBWG(l,(xyz[1],xyz[2]))
    else
        xyz[b[2] + 1] = T2((s[2] == 0) ? o.xyz[a[1] + 1] : g - o.xyz[a[2] + 1])
        return OctantBWG(l,(xyz[1],xyz[2],xyz[3]))
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

# Face indices permutation from p4est idx to Ferrite idx
const 𝒱₂_perm = [4
                 2
                 1
                 3]

# Face indices permutation from Ferrite idx to p4est idx
const 𝒱₂_perm_inv = [3
                     2
                     4
                     1]

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
