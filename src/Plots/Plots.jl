module Plots

using PyCall: PyNULL, pyimport
using PyPlot: isjulia_display, matplotlib
using Parameters: @unpack

# Pre-allocate Python bindings
const mpl         = PyNULL()    # Matplotlib
const plt         = PyNULL()    # PyPlot
const axes_grid1  = PyNULL()    # mpl_toolkits.axes_grid1
const backend_agg = PyNULL()    # mpl.backends.backend_agg
const tikzplotlib = PyNULL()    # Tikzplotlib

export mpl, plt

function __init__()
    # Bind Python libraries
    copy!(tikzplotlib, pyimport("tikzplotlib"))
    copy!(axes_grid1, pyimport("mpl_toolkits.axes_grid1"))
    copy!(mpl, matplotlib)
    copy!(plt, mpl.pyplot)
    copy!(backend_agg, mpl.backends.backend_agg)

    # Ensure not using Type 3 fonts
    plt.rc("pdf", fonttype=42)

    # Do not show figures automatically in IJulia
    isjulia_display[] = false
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
function plot(p::AbstractPlot, args...; kwargs...)
    fig, ax = plt.subplots()
    plot!(ax, p, args...; kwargs...)
    return fig
end

"""
    plot!([ax=plt.gca()], p::AbstractPlot)

Usage:
```julia
fig, ax = plt.subplots()
plot!(ax, p)
```
"""
plot!(ax, p::AbstractPlot) = throw(MethodError(plot!, p))
plot!(p::AbstractPlot, args...; kwargs...) = plot!(plt.gca(), p, args...; kwargs...)

"""
    get_tikz_code([fig], p::AbstractPlot; kwargs...)

Usage:
```julia
fig = plot(p)
code = get_tikz_code(fig, p)
```
"""
get_tikz_code(fig, p::AbstractPlot; kwargs...) = tikzplotlib.get_tikz_code(plot(p); kwargs...)
get_tikz_code(p::AbstractPlot; kwargs...) = get_tikz_code(plt.gcf(), p; kwargs...)

"""
    savefig([fig], p::AbstractPlot, fname::String)

Usage:
```julia
fig = plot(p)
savefig(fig, p)
```
"""
function savefig(fig, p::AbstractPlot, fname::String; kwargs...)
    ext = last(split(fname, "."))
    if ext == "tex"
        open(fname, "w") do io
            write(io, get_tikz_code(fig, p; kwargs...))
        end
    else
        fig.savefig(fname; kwargs...)
    end
end
savefig(p::AbstractPlot; kwargs...) = savefig(plt.gcf(), p; kwargs...)

export plot, plot!, get_tikz_code, savefig

### Ultilies

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
    return xlims, ylims, zlims
end

function autoset_lims!(ax, x)
    dim = size(x, 1)
    xlims, ylims, zlims = autoget_lims(x)
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    !isnothing(zlims) && ax.set_zlim(zlims)
end
autoset_lims!(x) = autoset_lims!(plt.gca(), x)

"""
    make_imggrid(img::AbstractArray{<:Number,4}, n_rows, n_cols; gap::Int=1)

Create a n_rows by n_cols image grid.
"""
function make_imggrid(img::AbstractArray{<:Number,4}, n_rows, n_cols; gap::Int=1)
    w, h, c, n = size(img)
    @assert c == 1 || c == 3
    imggrid = 0.5 * ones(n_rows * (w + gap) + gap, n_cols * (h + gap) + gap, c)
    i = 1
    for row = 1:n_rows, col = 1:n_cols
        if i <= n
            i_row = (row - 1) * (w + gap) + 1
            i_col = (col - 1) * (h + gap) + 1
            imggrid[i_row+1:i_row+w,i_col+1:i_col+h,:] .= img[:,:,:,i]
        else
            break
        end
        i += 1
    end
    if c == 1  # imshow supports gray as 3D tensor
        imggrid = dropdims(imggrid; dims=3)
    end
    return imggrid
end

function make_imggrid(img::AbstractArray{<:Number,3}, n_rows, n_cols; gap::Int=1)
    w, h, n = size(img)
    return make_imggrid(reshape(img, w, h, 1, n), n_rows, n_cols; gap=gap)
end

"""
    make_imggrid(img; gap::Int=1)

Create an approximately squared image grid based on the number of images.
For example, if last(size(img)) is 100, the grid will be 10 by 10.
"""
function make_imggrid(img; gap::Int=1)
    n = last(size(img))
    n_rows = ceil(Int, sqrt(n))
    n_cols = n_rows * (n_rows - 1) > n ? n_rows - 1 : n_rows
    return make_imggrid(img, n_rows, n_cols; gap=gap)
end

export autoget_lims, autoset_lims!, make_imggrid

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

export TwoYAxesLines, ImageGrid, ContourFunction, plot_actmat!

end # module
