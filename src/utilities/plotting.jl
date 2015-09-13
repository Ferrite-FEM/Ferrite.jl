const LTYPES = ["-", "--", ":"]
const LCOLORS = ["k", "b", "m", "r"]
const LMARKS = ["o", "*", ""]

"""
    eldraw2(ex, ey, [plotpar=[1,1,0], elnum=zeros(0)])

Draws the 2D mesh defined by ex, ey.
"""
function eldraw2(ex::VecOrMat, ey::VecOrMat,
                plotpar = [1, 1, 0], elnum=zeros(0))

    # TODO, Make it nice for elements with curved boundaries
    error_check_plotting(plotpar)

    ltype  = @compat Int(plotpar[1])
    lcolor = @compat Int(plotpar[2])
    lmark  = @compat Int(plotpar[3])
    lmark += 1

    plot_string = LTYPES[ltype] * LCOLORS[lcolor] * LMARKS[lmark]

    nx = size(ex, 2)
    ny = size(ey, 2)
    center = [sum(ex, 2)/nx sum(ey, 2)/ny]

    # TODO can't we make this a bit nicer /JB
    npoints = size(ex , 1)

    if npoints == 2
      xs = ex
      ys = ey
    else
      xs = [ex ex[:, 1]]'
      ys = [ey ey[:, 1]]'
    end
    p = winston().plot(xs, ys, plot_string)
    for el in elnum
         winston().text(center[el,1], center[el,2], string(el))
    end

    return p
end


"""
    eldisp2(ex, ey, ed, [plotpar=[1,1,0], sfac=1.0])

Draws the displaced 2D mesh defined by ex, ey and the displacements
given in ed.
"""
function eldisp2(ex::VecOrMat, ey::VecOrMat, ed::VecOrMat,
                plotpar = [1, 1, 0], sfac = 1.0)

    # TODO, Make it nice for elements with curved boundaries
    error_check_plotting(plotpar)

    ltype =  @compat Int(plotpar[1])
    lcolor = @compat Int(plotpar[2])
    lmark =  @compat Int(plotpar[3])
    lmark += 1

    plot_string = LTYPES[ltype] * LCOLORS[lcolor] * LMARKS[lmark]

    # TODO can't we make this a bit nicer /JB
    npoints = size(ex,2)
    if npoints == 2
      xs = ex + sfac * ed[:, 1:2:end]
      ys = ey + sfac * ed[:, 2:2:end]
    else
      xs = [ex + sfac * ed[:, 1:2:end] ex[:,1] + sfac * ed[:,1]]
      ys = [ey + sfac * ed[:, 2:2:end] ey[:,2] + sfac * ed[:,2]]
    end
    p = winston().plot(xs', ys', plot_string)

    return p
end

function error_check_plotting(plotpars)
    (@compat Int(plotpars[1])) in (1,2,3)   || throw(ArgumentError("linetype must be 1,2,3"))
    (@compat Int(plotpars[2])) in (1,2,3,4) || throw(ArgumentError("linecolor must be 1,2,3,4"))
    (@compat Int(plotpars[3])) in (1,2,0)   || throw(ArgumentError("linemark must be 1,2,0"))
end
