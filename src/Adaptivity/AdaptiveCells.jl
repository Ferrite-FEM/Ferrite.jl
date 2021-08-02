abstract type AbstractAdaptiveTree{dim,N,M} <: AbstractCell{dim,N,M} end

# Follow z order, x before y before z for faces, edges and corners
struct Octant{dim,N,M} <: AbstractAdaptiveTree{dim,N,M}
    nodes::Vector{Node} 
end

# return the two adjacent faces $f_i$ adjacent to edge `edge`
function _face(edge::Int)
    #maybe @view?
    return 𝒮[edge, :]
end

# return the `i`-th adjacent face fᵢ to edge `edge`
function _face(edge::Int, i::Int)
    return 𝒮[edge, i]
end

# return two face corners ξᵢ of the face `face` along edge `edge`
function _face_corners(edge::Int, face::Int) 
    #maybe @view
    return 𝒯[edge,face] 
end

# return the two `edge` corners cᵢ
function _edge_corners(edge::Int)
    #maybe @view
    return 𝒰[edge,:]
end

# return the `i`-th edge corner of `edge`
function _edge_corners(edge::Int,i::Int)
    return 𝒰[edge,i]
end

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

# finds face corner ξ′ in f′ for two associated faces f,f′ in {1,...,6} and their orientation r in {1,...,4}}
function _neighbor_corner(f,f′,r,ξ)
    return 𝒫[𝒬[ℛ[f,f′],r],ξ]
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
