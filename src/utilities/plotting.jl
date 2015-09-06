const LTYPES = ["-", "--", ":"]
const LCOLORS = ["k", "b", "m", "r"]
const LMARKS = ["o", "*", ""]

"Draws the 2D mesh defined by ex, ey."
function eldraw2(ex::AbstractVecOrMat, ey::AbstractVecOrMat,
                plotpar = [1, 1, 0], elnum::AbstractVector=zeros(0))

    # TODO, Make it nice for elements with curved boundaries
    error_check_plotting(plotpar)

    ltype =  Int(plotpar[1])
    lcolor = Int(plotpar[2])
    lmark =  Int(plotpar[3])
    lmark += 1

    plot_string = LTYPES[ltype] * LCOLORS[lcolor] * LMARKS[lmark]

    nnodes = size(ex, 1)
    center = [sum(ex, 1) / nnodes; sum(ey, 1) / nnodes]
    xs = [ex; ex[1,:]]
    ys = [ey; ex[1,:]]

    p = winston().plot(xs, ys, plot_string)
    for el in elnum
         winston().text(center[1, el], center[2, el], string(el))
    end

    return p
end


"Draws the displaced 2D mesh defined by ex, ey and the displacements
given in ed"
function eldisp2(ex::AbstractVecOrMat, ey::AbstractVecOrMat, ed::AbstractVecOrMat,
                plotpar = [1, 1, 0], sfac = 1.0)

    # TODO, Make it nice for elements with curved boundaries
    error_check_plotting(plotpar)

    ltype =  Int(plotpar[1])
    lcolor = Int(plotpar[2])
    lmark =  Int(plotpar[3])
    lmark += 1

    plot_string = LTYPES[ltype] * LCOLORS[lcolor] * LMARKS[lmark]


    xs = [ex + sfac * ed[:, 1:2:end]; ex[1,:] + sfac * ed[1, 1:2:end]]
    ys = [ey + sfac * ed[:, 1:2:end]; ey[1,:] + sfac * ed[1, 2:2:end]]

    p = winston().plot(xs, ys, plot_string)

    return p
end

function error_check_plotting(plotpars)
    Int(plotpars[1]) in (1,2,3)   || throw(ArgumentError("linetype must be 1,2,3"))
    Int(plotpars[2]) in (1,2,3,4) || throw(ArgumentError("linecolor must be 1,2,3,4"))
    Int(plotpars[3]) in (1,2,0)   || throw(ArgumentError("linemark must be 1,2,0"))
end