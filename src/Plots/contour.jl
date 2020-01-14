"""
A plot of 2D function.
"""
struct TwoDimFunction{T<:Function} <: AbstractPlot
    f::T
end

function plot!(
    ax, 
    p::TwoDimFunction, 
    xrange=ax.get_xlim(), 
    yrange=ax.get_ylim(); 
    contourevals=100, 
    alpha=0.3,
    kwargs...
)
    xbins = range(xrange..., length=contourevals)
    ybins = range(yrange..., length=contourevals)
    xygrid = [[xy...] for xy in Iterators.product(xbins, ybins)]
    xgrid = map(xy -> xy[1], xygrid)
    ygrid = map(xy -> xy[2], xygrid)
    zgrid = p.f(hcat(xygrid[:]...))
    zgrid = reshape(zgrid, size(xgrid)...)
    ax.contour(xgrid, ygrid, zgrid; alpha=alpha, kwargs...)
    return ax
end
