module Plots

using PyCall: PyNULL, pyimport
using PyPlot: PyPlot, isjulia_display, matplotlib
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
    plt.rc("pdf"; fonttype=42)

    # Use ggplot style
    plt.style.use("ggplot")

    # Do not show figures automatically in IJulia
    isjulia_display[] = false
end

### APIs

abstract type AbstractPlot end

"""
    plot(p::AbstractPlot, args...; figsize=nothing, kwargs...)

Usage:
```julia
fig = plot(p)
```
"""
function plot(p::AbstractPlot, args...; figsize=nothing, kwargs...)
    fig, ax = plt.subplots(; figsize=figsize)
    plot!(ax, p, args...; kwargs...)
    return fig
end

"""
    plot!([ax=plt.gca()], p::AbstractPlot, args...; kwargs...)

Usage:
```julia
fig, ax = plt.subplots()
plot!(ax, p)
```
"""
plot!(p::AbstractPlot, args...; kwargs...) = plot!(plt.gca(), p, args...; kwargs...)
plot!(ax, p::AbstractPlot) = throw(MethodError(plot!, p))

"""
    get_tikz_code([fig], p::AbstractPlot; kwargs...)

Usage:
```julia
fig = plot(p)
code = get_tikz_code(fig, p)
```
"""
get_tikz_code(fig::PyPlot.Figure, p::AbstractPlot; kwargs...) = tikzplotlib.get_tikz_code(plot(p); kwargs...)
get_tikz_code(p::AbstractPlot; kwargs...) = get_tikz_code(plt.gcf(), p; kwargs...)
function get_tikz_code(fig::PyPlot.Figure, p::Nothing=nothing; kwargs...)
    @warn "The TeX file by get_tikz_code(fig) may differ from Matplotlib's. Use get_tikz_code(fig, p) whenever possible."
    return tikzplotlib.get_tikz_code(fig; kwargs...)
end

"""
    savefig([fig], p::AbstractPlot, fname::String; bbox_inches="tight", kwargs...)

Usage:
```julia
fig = plot(p)
savefig(fig, p, "fig.png)
savefig(fig, p, "fig.tex)
```
"""
function savefig(fig::PyPlot.Figure, p::Union{AbstractPlot, Nothing}, fname::String; bbox_inches="tight", kwargs...)
    ext = last(split(fname, "."))
    if ext == "tex"
        open(fname, "w") do io
            write(io, get_tikz_code(fig, p; kwargs...))
        end
    else
        fig.savefig(fname; bbox_inches=bbox_inches, kwargs...)
    end
end
savefig(fig::PyPlot.Figure, fname::String; kwargs...) = savefig(fig, nothing, fname; kwargs...)
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
    make_imggrid(img::AbstractArray{<:Number,4}, nrows, ncols; gap::Int=1)

Create a nrows by ncols image grid.
"""
function make_imggrid(img::AbstractArray{<:Number,4}, nrows, ncols; gap::Int=1)
    w, h, c, n = size(img)
    @assert c == 1 || c == 3
    imggrid = 0.5 * ones(nrows * (w + gap) + gap, ncols * (h + gap) + gap, c)
    i = 1
    for row = 1:nrows, col = 1:ncols
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

function make_imggrid(img::AbstractArray{<:Number,3}, args...; kwargs...)
    w, h, n = size(img)
    img = reshape(img, w, h, 1, n)
    return make_imggrid(img, args...; kwargs...)
end

"""
    make_imggrid(img; kwargs...)

Create an approximately squared image grid based on the number of images.
For example, if last(size(img)) is 100, the grid will be 10 by 10.
"""
function make_imggrid(img; kwargs...)
    n = last(size(img))
    nrows = ceil(Int, sqrt(n))
    ncols = nrows * (nrows - 1) > n ? nrows - 1 : nrows
    return make_imggrid(img, nrows, ncols; kwargs...)
end

function count_leadingzeros(z::Vector)
    # Iterate until the frist non-zero item
    n = 1
    while n <= length(z) && z[n] == 0
        n = n + 1
    end
    return n - 1
end

export autoget_lims, autoset_lims!, make_imggrid, count_leadingzeros

### Plots

include("line.jl")
export TwoYAxesLines, OneDimFunction, LinesWithErrorBar
include("image.jl")
export ImageGrid, FeatureActivations
include("contour.jl")
export TwoDimFunction

using Distributions: ContinuousUnivariateDistribution, ContinuousMultivariateDistribution, logpdf

Plot(dist::ContinuousUnivariateDistribution)   = OneDimFunction(x -> exp.(logpdf(dist, x)))
Plot(dist::ContinuousMultivariateDistribution) = TwoDimFunction( x -> exp.(logpdf(dist, x)))

export Plot

end # module
