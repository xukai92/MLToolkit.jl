module Datasets

using Random: AbstractRNG, MersenneTwister

abstract type AbstractDataset{D} end

ratio2num(n_data, ratio) = floor(Int, n_data * ratio)

### Synthetic datasets

include("ring.jl")
export RingDataset, TwoDimRingDataset, ThreeDimRingDataset
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
    rng::AbstractRNG=GLOBAL_RNG,
) where {T}
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

include("vis.jl")
export n_display, vis
include("utilites.jl")
export Dataset

export Datasets

end # module
