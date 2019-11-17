import PGFPlots: plot, save
using PGFPlots: Axis, Plots, ColorMaps

# NOTE: function starting with `make_xxx_plot` will return a figure instance while
#       function starting with `plot_xxx` will return the axes being plotted on

### Two lines with shared y-axis

Parameters.@with_kw struct TwoYAxesLines{T<:AbstractVector{<:AbstractFloat}}
    x::T
    y1::T
    y2::T
    mark="none"
    colour1="red"
    colour2="blue"
    xlabel=nothing
    ylabel1=nothing
    ylabel2=nothing
end

"""
Reference: https://latex.org/forum/viewtopic.php?t=21317
"""
function plot(p::TwoYAxesLines)
    a1 = Axis(Plots.Linear(p.x, p.y1; mark=p.mark, style="color=$(p.colour1)"); xlabel=p.xlabel, ylabel=p.ylabel1)
    a2 = Axis(Plots.Linear(p.x, p.y2; mark=p.mark, style="color=$(p.colour2)"); axisLines="right", ylabel=p.ylabel2, ylabelStyle="rotate=180")
    return [a1, a2]
end

### Images

struct GrayImages{T<:AbstractArray{<:AbstractFloat,3}}
    imgs::T
end

"""
    make_imggrid(gimgs::GrayImages, n_rows::Int, n_cols::Int; gap::Int=1)

Create a `n_rows` by `n_cols` image grid using `gimgs`.

NOTE: only gray images are supported at the moment.
"""
function make_imggrid(gimgs::GrayImages, n_rows, n_cols; gap::Int=1)
    d_row, d_col, n = size(gimgs.imgs)
    imggrid = 0.5 * ones(n_rows * (d_row + gap) + gap, n_cols * (d_col + gap) + gap)
    i = 1
    for row = 1:n_rows, col = 1:n_cols
        if i <= n
            i_row = (row - 1) * (d_row + gap) + 1
            i_col = (col - 1) * (d_col + gap) + 1
            imggrid[i_row+1:i_row+d_row,i_col+1:i_col+d_col] .= gimgs.imgs[:,:,i]
        else
            break
        end
        i += 1
    end
    return imggrid
end

make_imggrid(imgs, n_rows, n_cols; gap::Int=1) = make_imggrid(GrayImages(imgs), n_rows, n_cols; gap=gap)

function GrayImages(imgs::AbstractMatrix{<:AbstractFloat})
    dsq = size(imgs, 1)
    try
        d = convert(Int, sqrt(dsq))
        imgs = reshape(imgs, d, d, last(size(imgs)))
    catch
        @error "Cannot automatically convert an image which is not sqaured."
    end
    return GrayImages(imgs)
end

function GrayImages(imgs::AbstractMatrix{<:AbstractFloat}, shape::Tuple{Int,Int})
    imgs = reshape(imgs, last(size(imgs)), shape...)
    return GrayImages(imgs)
end

function GrayImages(imgs::AbstractArray{<:AbstractFloat,4})
    imgs = dropdims(imgs; dims=3)
    return GrayImages(imgs)
end

function plot(gimgs::GrayImages, n_rows::Int, n_cols::Int)
    imggrid = make_imggrid(gimgs, n_rows, n_cols)
    return Axis(Plots.Image(imggrid, (1, size(imggrid, 2)), (1, size(imggrid, 1)); colormap=ColorMaps.GrayMap(invert=false)); axisEqualImage=true)
end

function plot(gimgs::GrayImages)
    n = last(size(gimgs.imgs))
    n_rows = ceil(Int, sqrt(n))
    n_cols = n_rows * (n_rows - 1) > n ? n_rows - 1 : n_rows
    return plot(gimgs, n_rows, n_cols)
end

function plot!(gimgs::GrayImages, n_rows::Int, n_cols::Int; ax=plt.gca())
    im = ax."imshow"(make_imggrid(gimgs, n_rows, n_cols), cmap="gray")
    plt.axis("off")
    divider = axes_grid1.make_axes_locatable(ax)
    cax = divider."append_axes"("bottom", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax, orientation="horizontal")
    return ax
end

# TODO: merge GrayImages and RGBImages
struct RGBImages{T<:AbstractArray{<:AbstractFloat,4}}
    imgs::T
end

function make_imggrid(imgs::RGBImages, n_rows, n_cols; gap::Int=1)
    d_row, d_col, n_channels, n = size(imgs.imgs)
    @assert n_channels == 3
    imggrid = 0.5 * ones(n_rows * (d_row + gap) + gap, n_cols * (d_col + gap) + gap, 3)
    i = 1
    for row = 1:n_rows, col = 1:n_cols
        if i <= n
            i_row = (row - 1) * (d_row + gap) + 1
            i_col = (col - 1) * (d_col + gap) + 1
            imggrid[i_row+1:i_row+d_row,i_col+1:i_col+d_col,:] .= imgs.imgs[:,:,:,i]
        else
            break
        end
        i += 1
    end
    return imggrid
end

function plot!(imgs::RGBImages, n_rows::Int, n_cols::Int; ax=plt.gca())
    im = ax."imshow"(make_imggrid(imgs, n_rows, n_cols))
    plt.axis("off")
    return ax
end

# Automatic decide the rows and cols based on batch size
function plot!(imgs::Union{GrayImages, RGBImages}; ax=plt.gca())
    n = last(size(imgs.imgs))
    n_rows = ceil(Int, sqrt(n))
    n_cols = n_rows * (n_rows - 1) > n ? n_rows - 1 : n_rows
    return plot!(imgs, n_rows, n_cols; ax=ax)
end

### Contour

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
    return plot(ContourFunction(x -> exp.(logpdf(dist, x))), xrange, yrange; contourevals=100)
end

# function make_two_y_axes_plot(xs, ys1, ys2; color1="tab:red", color2="tab:blue",
#                               xlabel=nothing, ylabel1=nothing, ylabel2=nothing)
#     fig, ax1 = plt.subplots()
#     ax1."plot"(xs, ys1, c=color1)
#     xlabel == nothing || ax1."set_xlabel"(xlabel)
#     ylabel1 == nothing || ax1."set_ylabel"(ylabel1, color=color1)
#     ax1."tick_params"(axis="y", labelcolor=color1)
#     ax2 = ax1."twinx"()
#     ax2."plot"(xs, ys2, c=color2)
#     ylabel2 == nothing || ax2."set_ylabel"(ylabel2, color=color2)
#     ax2."tick_params"(axis="y", labelcolor=color2)
#     fig."tight_layout"()
#     return fig
# end
#
# function plot_grayimg!(img, args...; ax=plt.gca())
#     @assert length(args) == 0 || length(args) == 2 "You can either plot a single image or declare the `n_rows` and `n_cols`"
#     im = ax."imshow"(make_imggrid(img, args...), cmap="gray")
#     plt.axis("off")
#     divider = axes_grid1.make_axes_locatable(ax)
#     cax = divider."append_axes"("bottom", size="5%", pad=0.05)
#     plt.colorbar(im, cax=cax, orientation="horizontal")
#     return ax
# end

function plot_actmat!(Z::Matrix; ax=plt.gca())
    # TODO: implement a sorting version
    # col_sort_idcs = sortperm(vec([count_leadingzeros(Z[:,k]) for k = 1:size(Z, 2)]))
    # Z = Z[:,col_sort_idcs]
    ax."imshow"(Z, cmap="Greys", interpolation="nearest", vmin=0.0, vmax=1.0)
    ax.set_xticks([])
    ax.set_yticks([])
    return ax
end

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
# plot_pdf!(dist::Distributions.ContinuousMultivariateDistribution) = plot_contour!(x -> exp.(logpdf(dist, x)))
