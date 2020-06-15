module PlotRecipes

using UnPack, Plots
import FileIO: save

save(fn::String, p::AbstractPlot) = savefig(p, fn)

export save

### Utilities

function autoget_lims(x)
    xlims = [extrema(x[1,:])...]
    dx = xlims[2] - xlims[1]
    xlims += [-0.1dx, +0.1dx]
    ylims = [extrema(x[2,:])...]
    dy = ylims[2] - ylims[1]
    ylims += [-0.1dy, +0.1dy]
    if size(x, 1) == 3
        zlims = [extrema(x[3,:])...]
        dz = zlims[2] - zlims[1]
        zlims += [-0.1dz, +0.1dz]
    else
        zlims = nothing
    end
    return (xlims=xlims, ylims=ylims, zlims=zlims)
end

function autoset_lims!(p, x)
    dim = size(x, 1)
    xlims, ylims, zlims = autoget_lims(x)
    xlims!(p, xlims)
    ylims!(p, ylims)
    !isnothing(zlims) && zlims!(p, zlims)
end
autoset_lims!(x) = autoset_lims!(current(), x)

export autoget_lims, autoset_lims!

### Plots

include("line.jl")
export ErrorBarLines, TwoYAxesLines
include("function.jl")
export OneDimFunction, TwoDimFunction
include("scatter.jl")
export LabelledScatter
include("image.jl")
export ImageGrid, FeatureActivations, make_imggrid, count_leadingzeros
include("path.jl")
export TwoDimPath

end # module
