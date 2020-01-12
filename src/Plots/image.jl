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

function plot!(ax, p::ImageGrid, args::Int...; gap::Int=1)
    imggrid = make_imggrid(p.img, args...; gap=gap)
    if length(size(imggrid)) == 2
        im = ax.imshow(imggrid, cmap="gray")
        divider = axes_grid1.make_axes_locatable(ax)
        cax = divider.append_axes("bottom", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax, orientation="horizontal")
    else
        im = ax.imshow(imggrid)
    end
    plt.axis("off")
    return ax
end
