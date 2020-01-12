"""
A plot of images in grid.
"""
struct ImageGrid{T<:AbstractArray{<:Number}} <: AbstractPlot
    img::T
end

function ImageGrid(img::AbstractMatrix{<:Number})
    dsq = size(img, 1)
    try
        d = convert(Int, sqrt(dsq))
        return ImageGrid(img, (d, d))
    catch
        @error "Cannot automatically convert an image which is not squared."
    end
end

function ImageGrid(img::AbstractMatrix{<:Number}, shape::Tuple{Vararg{Int,N}}) where {N}
    @assert N == 2 || N == 3
    img = reshape(img, shape..., last(size(img)))
    return ImageGrid(img)
end

function plot!(ax, p::ImageGrid, args::Int...; kwargs...)
    imggrid = make_imggrid(p.img, args...; kwargs...)
    if length(size(imggrid)) == 2
        im = ax.imshow(imggrid; cmap="gray")
        divider = axes_grid1.make_axes_locatable(ax)
        cax = divider.append_axes("bottom"; size="5%", pad=0.05)
        plt.colorbar(im; cax=cax, orientation="horizontal")
    else
        im = ax.imshow(imggrid)
    end
    ax.set_xticks([])
    ax.set_yticks([])
    return ax
end

struct FeatureActivation <: AbstractPlot
    Z::Matrix{Bool}
end

function plot!(ax, p::FeatureActivation)
    # TODO: implement a sorting version
    # col_sort_idcs = sortperm(vec([count_leadingzeros(Z[:,k]) for k = 1:size(Z, 2)]))
    # Z = Z[:,col_sort_idcs]
    ax.imshow(p.Z; cmap="gray", interpolation="nearest", vmin=0.0, vmax=1.0)
    ax.set_xticks([])
    ax.set_yticks([])
    return ax
end
