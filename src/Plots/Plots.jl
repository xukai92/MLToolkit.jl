module Plots

using PyCall: PyNULL, pyimport
using PyPlot: isjulia_display, matplotlib
using Parameters: @with_kw, @unpack

# Pre-allocate Python bindings
const mpl         = PyNULL()    # Matplotlib
const plt         = PyNULL()    # PyPlot
const axes_grid1  = PyNULL()    # mpl_toolkits.axes_grid1
const backend_agg = PyNULL()    # mpl.backends.backend_agg
const tikzplotlib = PyNULL()    # Tikzplotlib

export mpl, plt

function __init__()
    # Do not show figures automatically in IJulia
    isjulia_display[] = false
    # Bind Python libraries
    copy!(tikzplotlib, pyimport("tikzplotlib"))
    copy!(axes_grid1, pyimport("mpl_toolkits.axes_grid1"))
    copy!(mpl, matplotlib)
    copy!(plt, mpl.pyplot)
    copy!(backend_agg, mpl.backends.backend_agg)
    # Ensure not using Type 3 fonts
    plt.rc("pdf", fonttype=42)
end

### APIs

abstract type AbstractPlot end

"""
    plot(p::AbstractPlot)

Usage:
```julia
fig = plot(p)
```
"""
plot(p::AbstractPlot)  = throw(MethodError(plot,  p))

"""
    plot!([ax=plt.gca()], p::AbstractPlot)

Usage:
```julia
fig, ax = plt.subplots()
plot!(ax, p)
```
"""
plot!(ax, p::AbstractPlot) = throw(MethodError(plot!, p))
plot!(p::AbstractPlot) = plot!(plt.gca(), p)

"""
    get_tikz_code(p::AbstractPlot)

Usage:
```julia
code = get_tikz_code(p)
```
"""
get_tikz_code(p::AbstractPlot) = tikzplotlib.get_tikz_code(plot(p))

export plot, plot!, get_tikz_code

### Ultilies

function autoset_lim!(x; ax=plt.gca())
    xlims = [extrema(x[1,:])...]
    dx = xlims[2] - xlims[1]
    xlims += [-0.1dx, +0.1dx]
    ylims = [extrema(x[2,:])...]
    dy = ylims[2] - ylims[1]
    ylims += [-0.1dy, +0.1dy]
    dim = size(x, 1)
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    if size(x, 1) == 3
        zlims = [extrema(x[3,:])...]
        dz = zlims[2] - zlims[1]
        zlims += [-0.1dz, +0.1dz]
        ax.set_zlim(zlims)
    end
end

export autoset_lim!

### Plots

include("line.jl")
include("image.jl")
# include("contour.jl")

function plot_actmat!(Z::Matrix; ax=plt.gca())
    # TODO: implement a sorting version
    # col_sort_idcs = sortperm(vec([count_leadingzeros(Z[:,k]) for k = 1:size(Z, 2)]))
    # Z = Z[:,col_sort_idcs]
    ax."imshow"(Z, cmap="Greys", interpolation="nearest", vmin=0.0, vmax=1.0)
    ax.set_xticks([])
    ax.set_yticks([])
    return ax
end

export TwoYAxesLines, GrayImages, RGBImages, ContourFunction, plot_actmat!

end # module
