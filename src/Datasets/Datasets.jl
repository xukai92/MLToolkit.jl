module Datasets

using Random: MersenneTwister

abstract type AbstractDataset{D} end

ratio2num(n_data, ratio) = floor(Int, n_data * ratio)

### Synthetic datasets

include("ring.jl")
export RingDataset
include("gaussian.jl")
export GaussianDataset

### Image datasets

using MLDataUtils: convertlabel, LabelEnc
include("broadcasted_logit.jl")

abstract type ImageDataset{T, D} <: AbstractDataset{D} end

flatten(X) = reshape(X, :, last(size(X)))

function dequantize(rng, x, alpha::T) where {T}
    y = x + rand(rng, T) / 256
    y = clamp(y, 0, 1)
    y = alpha + (1 - 2alpha) * y
    return y
end

link_bin(x) = BroadcastedLogit(zero(eltype(x)), one(eltype(x)))(x)
invlink(d::ImageDataset{Val{:true}}, x) = inv(BroadcastedLogit(zero(eltype(x)), one(eltype(x))))(x)
invlink(d::ImageDataset{Val{:false}}, x) = x

function preprocess(rng, X, is_flatten, alpha, is_link::Bool=false)
    if is_flatten
        X = flatten(X)
    end
    if !iszero(alpha)
        X = dequantize.(Ref(rng), X, alpha)
    end
    if is_link
        X = link_bin(X)
    end
    return X
end

# TODO: implement flip augmentation
# Ref: https://github.com/gpapamak/maf/blob/master/datasets/cifar10.py#L52-L62
function get_image_data(
    IMAGE,
    n_data::Int,
    n_test::Int,
    is_flatten::Bool,
    alpha::T, 
    is_link::Bool,
    K::Int,
    perm::Union{NTuple{3,Int}, NTuple{4,Int}};
    seed::Int=1,
) where {T}
    rng = MersenneTwister(seed)
    onehot_enc = LabelEnc.OneOfK(K)

    function preprocess_tuple(X, y)
        X = preprocess(rng, X, is_flatten, alpha, is_link)
        Y = convertlabel(onehot_enc, y)
        return X, y, Y
    end

    X = permutedims(IMAGE.traintensor(T, 1:n_data), perm)
    y = IMAGE.trainlabels(1:n_data) .+ 1
    X, y, Y = preprocess_tuple(X,  y)

    Xt = permutedims(IMAGE.testtensor(T, 1:n_test), perm)
    yt = IMAGE.testlabels(1:n_test) .+ 1
    Xt, yt, Yt = preprocess_tuple(Xt, yt)
    
    return X, y, Y, Xt, yt, Yt
end

include("mnist.jl")
export MNISTDataset
include("cifar10.jl")
export CIFAR10Dataset
include("features.jl")
export FeatureDataset, get_features_griffiths2011, get_features_xu2019

### Visualizations

using ..MLToolkit.Plots

function vis(dataset::AbstractDataset, args...; kwargs...)
    fig, ax = plt.subplots(figsize=(5, 5))
    vis!(ax, dataset, args...; kwargs...)
    return fig
end

function vis(dataset::AbstractDataset{3}, args...; kwargs...)
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(projection="3d")
    vis!(ax, dataset, args...; kwargs...)
    return fig
end

vis!(ax, d::AbstractDataset, x) = vis!(ax, d, (x=x,))

function vis!(ax, ::Union{AbstractDataset{2}, AbstractDataset{3}}, nt::NamedTuple)
    alpha = length(nt) > 0.75 ? 0.5 : 1.0
    for (x, label) in zip(values(nt), keys(nt))
        ax.scatter([x[i,:] for i in 1:size(x, 1)]..., marker=".", alpha=alpha, label=label)
    end
    autoset_lims!(ax, first(values(nt)))
    length(nt) > 1 && ax.legend(fancybox=true, framealpha=0.5)
end

function vis!(ax, d::ImageDataset, nt::NamedTuple{T1, <:NTuple{N, T2}}) where {T1, N, T2}
    plot!(ax, ImageGrid(invlink(d, cat(values(nt)...; dims=ndims(T2)))))
end

export Datasets, n_display, vis

end # module
