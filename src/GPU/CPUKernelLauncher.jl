function init_kernel(::Type{BackendCPU}, n_cells::Ti, n_basefuncs::Ti, kernel::Function, args::Tuple) where {Ti <: Integer}
    return LazyKernel(n_cells, n_basefuncs, kernel, args, BackendCPU)
end

function launch!(kernel::LazyKernel{Ti, BackendCPU}) where {Ti}
    ker = kernel.kernel
    args = kernel.args
    ## Naive implementation to circumvent the issue with cellvalues
    ## on GPU the we are using the static version of cellvalues because it's immutable
    ## so in order to unify the parallel kernel interface we need (for now) to use the static version
    ## without changing the routine, so basically we search for any cellvalues passed in the args and
    ## convert it to the static version
    cell_index = findfirst(x -> x isa CellValues, args)
    (cell_index === nothing) || (args = _update_cell_args(args, cell_index))
    args, color_dh = _to_colordh(args) # convert the dofhandler to color dofhandler
    no_colors = ncolors(color_dh)
    nthreads = Threads.nthreads()
    for i in 1:no_colors
        current_color!(color_dh, i)
        Threads.@threads :static for j in 1:nthreads
            ker(args...)
        end
    end
    return
end


function _to_colordh(args::Tuple)
    dh_index = findfirst(x -> x isa AbstractDofHandler, args)
    dh_index !== nothing || throw(ErrorException("No subtype of AbstractDofHandler found in the arguments"))
    arr = args |> collect
    color_dh = init_colordh(arr[dh_index])
    arr[dh_index] = color_dh
    return Tuple(arr), color_dh
end

function _update_cell_args(args::Tuple, index::Int)
    ## since tuples are immutable we need to convert it to an array to update the values
    ## then convert it back to a tuple
    arr = args |> collect
    arr[index] = _to_static_cellvalues(arr[index])
    return Tuple(arr)
end


function _to_static_cellvalues(cv::CellValues)
    fv = StaticInterpolationValues(cv.fun_values)
    gm = StaticInterpolationValues(cv.geo_mapping)
    weights = ntuple(i -> getweights(cv.qr)[i], getnquadpoints(cv))
    return StaticCellValues(fv, gm, weights)
end
