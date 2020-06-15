### ImageGrid

using Colors: RGB

struct ImageGrid{T<:AbstractArray{<:Number}}
    data::T
end

function ImageGrid(data::AbstractMatrix{<:Number})
    local shape
    try
        d = size(data, 1)
        if d == 28 * 28         # MNIST-like
            shape = (28, 28)
        elseif d == 32 * 32 * 3 # CIFAR-like
            shape = (32, 32, 3)
        else
            l = convert(Int, sqrt(d))
            shape = (l, l)
        end
    catch
        @error "Cannot automatically convert an image which is not squared."
    end
    return ImageGrid(data, shape)
end

function ImageGrid(data::AbstractMatrix{<:Number}, shape::Tuple{Vararg{Int,N}}) where {N}
    @assert N == 2 || N == 3
    data = reshape(data, shape..., last(size(data)))
    return ImageGrid(data)
end

image(data::AbstractMatrix) = Matrix{RGB}(data)

function image(data::AbstractArray)
    w, h, c = size(data)
    img = Matrix{RGB}(undef, w, h)
    for i in 1:w, j in 1:h
        img[i,j] = RGB(data[i,j,:]...)
    end
    return img
end

"""
    make_imggrid(data::AbstractArray{<:Number,4}, nrows, ncols; gap::Int=1)

Create a nrows by ncols image grid.
"""
function make_imggrid(data::AbstractArray{<:Number,4}, nrows, ncols; gap::Int=1)
    w, h, c, n = size(data)
    @assert c == 1 || c == 3
    imggrid = 0.5 * ones(nrows * (w + gap) + gap, ncols * (h + gap) + gap, c)
    i = 1
    for row = 1:nrows, col = 1:ncols
        if i <= n
            i_row = (row - 1) * (w + gap) + 1
            i_col = (col - 1) * (h + gap) + 1
            imggrid[i_row+1:i_row+w,i_col+1:i_col+h,:] .= data[:,:,:,i]
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

function make_imggrid(data::AbstractArray{<:Number,3}, args...; kwargs...)
    w, h, n = size(data)
    data = reshape(data, w, h, 1, n)
    return make_imggrid(data, args...; kwargs...)
end

"""
    make_imggrid(data; kwargs...)

Create an approximately squared image grid based on the number of images.
For example, if last(size(data)) is 100, the grid will be 10 by 10.
"""
function make_imggrid(data; kwargs...)
    n = last(size(data))
    nrows = ceil(Int, sqrt(n))
    ncols = nrows * (nrows - 1) > n ? nrows - 1 : nrows
    return make_imggrid(data, nrows, ncols; kwargs...)
end

@recipe function f(ig::ImageGrid, args::Int...)
    imggrid = make_imggrid(ig.data, args...)
    
    seriestype --> :image
    
    image(imggrid)
end

### FeatureActivations

struct FeatureActivations{T<:AbstractMatrix{Bool}}
    Z::T
end

function FeatureActivations(Z::Matrix{Int})
    @assert Set(unique(Z)) == Set((0, 1))
    return FeatureActivations(Matrix{Bool}(Z))
end

function count_leadingzeros(z::Vector)
    # Iterate until the frist non-zero item
    n = 1
    while n <= length(z) && z[n] == 0
        n = n + 1
    end
    return n - 1
end

@recipe function f(fa::FeatureActivations; sort=false)
    Z = fa.Z
    
    if sort
        col_sort_idcs = sortperm(vec([count_leadingzeros(Z[:,k]) for k = 1:size(Z, 2)]))
        Z = Z[:,col_sort_idcs]
    end
    
    image(Z)
end
