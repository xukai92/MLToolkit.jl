using MLDatasets: MNIST
using MLDataUtils: convertlabel, LabelEnc

include("broadcasted_logit.jl")

struct MNISTDataset <: AbstractDataset
    X
    Y
    invlink
end

n_display(::MNISTDataset) = 100

function dequantize(rng, x, alpha::T) where {T}
    y = x + rand(rng, T) / 256
    y = clamp(y, 0, 1)
    y = alpha + (1 - 2alpha) * y
    return y
end

function MNISTDataset(
    n_data::Int; 
    seed::Int=1, 
    is_flatten::Bool=true, 
    alpha::T=0f0, 
    is_link::Bool=false,
    is_onehot::Bool=false,
) where {T}
    rng = MersenneTwister(seed)
    X = MNIST.traintensor(T, 1:n_data)
    if is_flatten
        X = reshape(permutedims(X, (2, 1, 3)), 784, :)
    end
    if !iszero(alpha)
        X = dequantize.(Ref(MersenneTwister(seed)), X, alpha)
    end
    if is_link
        link = BroadcastedLogit(0f0, 1f0)
        X = link(X)
        invlink = inv(link)
    else
        invlink(x) = x
    end
    Y = convertlabel(LabelEnc.OneOfK(10), MNIST.trainlabels() .+ 1)
    return MNISTDataset(X, Y, invlink)
end
