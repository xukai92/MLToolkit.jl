module Datasets

using Random: MersenneTwister

abstract type AbstractDataset{D} end

ratio2num(n_data, ratio) = Int(div(n_data, div(1, ratio)))

### Synthetic datasets

include("ring.jl")
export RingDataset
include("gaussian.jl")
export GaussianDataset

### Image datasets

include("broadcasted_logit.jl")

abstract type ImageDataset{T, D} <: AbstractDataset{D} end

function dequantize(rng, x, alpha::T) where {T}
    y = x + rand(rng, T) / 256
    y = clamp(y, 0, 1)
    y = alpha + (1 - 2alpha) * y
    return y
end

link_bin(x) = BroadcastedLogit(zero(eltype(x)), one(eltype(x)))(x)
invlink(d::ImageDataset{Val{:true}}, x) = inv(BroadcastedLogit(zero(eltype(x)), one(eltype(x))))(x)
invlink(d::ImageDataset{Val{:false}}, x) = x

function preprocess(rng, X, alpha, is_link)
    if !iszero(alpha)
        X = dequantize.(Ref(rng), X, alpha)
    end
    if is_link
        X = link_bin(X)
    end
    return X
end

include("mnist.jl")
export MNISTDataset
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
    alpha = length(nt) > 1 ? 0.5 : 1.0
    for (x, label) in zip(values(nt), keys(nt))
        ax.scatter([x[i,:] for i in 1:size(x, 1)]..., marker=".", alpha=alpha, label=label)
    end
    autoset_lims!(ax, first(values(nt)))
    length(nt) > 1 && ax.legend(fancybox=true, framealpha=0.5)
end

function vis!(ax, d::ImageDataset, nt::NamedTuple)
    plot!(ax, ImageGrid(invlink(d, hcat(values(nt)...))))
end

export Datasets, n_display, vis

end # module
