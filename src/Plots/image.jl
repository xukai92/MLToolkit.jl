# Image

struct GrayImages{T<:AbstractArray{<:Number,3}}
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

# function plot_grayimg!(img, args...; ax=plt.gca())
#     @assert length(args) == 0 || length(args) == 2 "You can either plot a single image or declare the `n_rows` and `n_cols`"
#     im = ax."imshow"(make_imggrid(img, args...), cmap="gray")
#     plt.axis("off")
#     divider = axes_grid1.make_axes_locatable(ax)
#     cax = divider."append_axes"("bottom", size="5%", pad=0.05)
#     plt.colorbar(im, cax=cax, orientation="horizontal")
#     return ax
# end

function GrayImages(imgs::AbstractMatrix{<:Number})
    dsq = size(imgs, 1)
    try
        d = convert(Int, sqrt(dsq))
        imgs = reshape(imgs, d, d, last(size(imgs)))
    catch
        @error "Cannot automatically convert an image which is not sqaured."
    end
    return GrayImages(imgs)
end

function GrayImages(imgs::AbstractMatrix{<:Number}, shape::Tuple{Int,Int})
    imgs = reshape(imgs, last(size(imgs)), shape...)
    return GrayImages(imgs)
end

function GrayImages(imgs::AbstractArray{<:Number,4})
    imgs = dropdims(imgs; dims=3)
    return GrayImages(imgs)
end

# function plot(gimgs::GrayImages, n_rows::Int, n_cols::Int)
#     imggrid = make_imggrid(gimgs, n_rows, n_cols)
#     return Axis(Plots.Image(imggrid, (1, size(imggrid, 2)), (1, size(imggrid, 1)); colormap=ColorMaps.GrayMap(invert=false)); axisEqualImage=true)
# end

# function plot(gimgs::GrayImages)
#     n = last(size(gimgs.imgs))
#     n_rows = ceil(Int, sqrt(n))
#     n_cols = n_rows * (n_rows - 1) > n ? n_rows - 1 : n_rows
#     return plot(gimgs, n_rows, n_cols)
# end

function plot!(gimgs::GrayImages, n_rows::Int, n_cols::Int; ax=plt.gca())
    im = ax."imshow"(make_imggrid(gimgs, n_rows, n_cols), cmap="gray")
    plt.axis("off")
    divider = axes_grid1.make_axes_locatable(ax)
    cax = divider."append_axes"("bottom", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax, orientation="horizontal")
    return ax
end

# TODO: merge GrayImages and RGBImages
struct RGBImages{T<:AbstractArray{<:Number,4}}
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
