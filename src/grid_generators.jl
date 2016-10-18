# 1D rectangle
function generate_grid{T}(nel::NTuple{1, Int}, left::Vec{1, T}=Vec{1}((-1.0,)), right::Vec{1, T}=Vec{1}((1.0,)))
    x_left = left[1]
    x_right = right[1]
    nel_x = nel[1]
    nel = nel_x
    n_nodes_x = nel_x + 1
    n_nodes = n_nodes_x

    coords_x = collect(linspace(x_left, x_right, n_nodes))

    # Generate nodes
    nodes = Node{1,T}[]
    for i in 1:n_nodes
        push!(nodes, Node{1,T}(Vec{1, T}((coords_x[i],))))
    end

    cells = Cell{1,2}[]
    for i in 1:nel_x
        push!(cells, Cell{1,2}(i, (i, i+1)))
    end

    # Do something with the boundary here, generate cell sets

    return Grid(cells, nodes)
end

# 2D rectangle
function generate_grid{T}(nel::NTuple{2, Int}, left::Vec{2, T}=Vec{2}((-1.0,-1.0)), right::Vec{2, T}=Vec{2}((1.0,1.0)))

    x_left = left[1]; y_left = left[2]
    x_right = right[1]; y_right = right[2]
    nel_x = nel[1]; nel_y = nel[2]
    nel = nel_x * nel_y
    n_nodes_x = nel_x + 1; n_nodes_y = nel_y + 1
    n_nodes = n_nodes_x * n_nodes_y

    coords_x = linspace(x_left, x_right, n_nodes_x)
    coords_y = linspace(y_left, y_right, n_nodes_y)

    # Generate nodes
    nodes = Node{2,T}[]
    for j in 1:n_nodes_y, i in 1:n_nodes_x
        push!(nodes, Node{2,T}(Vec{2, T}((coords_x[i],coords_y[j]))))
    end

    node_array = reshape(collect(1:n_nodes),(n_nodes_x,n_nodes_y))

    cells = Cell{2,4}[]
    n = 0
    for j in 1:nel_y, i in 1:nel_x
        n += 1
        push!(cells, Cell{2,4}(n, (node_array[i,j], node_array[i+1,j], node_array[i+1,j+1], node_array[i, j+1])))
    end

    # Do something with the boundary here, generate cell sets probably

    return Grid(cells, nodes)
end
