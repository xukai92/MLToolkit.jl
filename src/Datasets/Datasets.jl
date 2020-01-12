module Datasets

import Distributions: rand, logpdf
export rand, logpdf

using Random: AbstractRNG
using Distributions: ContinuousMultivariateDistribution, MixtureModel, MvNormal

"""
2D ring dataset.
"""
struct RingDataset{T<:AbstractFloat} <: ContinuousMultivariateDistribution
    mixturemodel
end

function RingDataset(n_clusters::Int, s::T, σ::T) where {T<:AbstractFloat}
    π_typed = convert(T, π)
    cluster_indices = collect(0:n_clusters-1)
    base_angle = π_typed * 2 / n_clusters
    angle = (base_angle .* cluster_indices) .- π_typed / 2
    μ = [s * cos.(angle) s * sin.(angle)]'
    return RingDataset{T}(MixtureModel([MvNormal(μ[:,i], σ) for i in 1:size(μ, 2)]))
end

rand(rng::AbstractRNG, dataset::RingDataset{T}, n::Int) where {T} = convert.(T, rand(rng, dataset.mixturemodel, n))

logpdf(dataset::RingDataset, x::AbstractArray{<:AbstractFloat,2}) = logpdf(dataset.mixturemodel, x)

export RingDataset

"""
Feature dataset.
"""
struct FeatureDataset{T} <: ContinuousMultivariateDistribution
    features::Matrix{T}
end

function rand(rng::AbstractRNG, dataset::FeatureDataset{T}, n::Int) where {T}
    n_features = size(dataset.features, 2)
    activation_matrix = rand(n_features, n) .> 0.5
    return dataset.features * activation_matrix
end

export FeatureDataset

include("features.jl")  # actual features
export get_features_griffiths2011, get_features_xu2019

end # module
