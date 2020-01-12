# Contour

import Distributions

struct ContourFunction
    f
end

function Contour_by_batchf(f, xrange, yrange, contourevals)
    xbins = range(xrange..., length=contourevals)
    ybins = range(yrange..., length=contourevals)
    xygrid = [[xy...] for xy in Iterators.product(xbins, ybins)]
    xgrid = map(xy -> xy[1], xygrid)
    ygrid = map(xy -> xy[2], xygrid)
    zgrid = f(hcat(xygrid[:]...))
    zgrid = reshape(zgrid, size(xgrid)...)
    return Plots.Contour(zgrid, xbins, ybins)
end

function plot(c::ContourFunction, xrange, yrange; contourevals=100)
    p = Contour_by_batchf(c.f, xrange, yrange, contourevals)
    return Axis(p; axisEqualImage=true)
end

function plot(dist::Distributions.ContinuousMultivariateDistribution, xrange, yrange; contourevals=100)
    return plot(ContourFunction(x -> exp.(Distributions.logpdf(dist, x))), xrange, yrange; contourevals=100)
end

# function plot_contour!(f; contourevals=100, alpha=0.3, ax=plt.gca())
#     rangex = range(ax.get_xlim()..., length=contourevals)
#     rangey = range(ax.get_ylim()..., length=contourevals)
#     gridxy = [[xy...] for xy in Iterators.product(rangex, rangey)]
#     gridx = map(xy -> xy[1], gridxy)
#     gridy = map(xy -> xy[2], gridxy)
#     gridz = f(hcat(gridxy[:]...))
#     gridz = reshape(gridz, size(gridx)...)
#     ax.contour(gridx, gridy, gridz, alpha=alpha)
# end
#
# plot_pdf!(dist::Distributions.ContinuousMultivariateDistribution) = plot_contour!(x -> exp.(Distributions.logpdf(dist, x)))
