module Datasets

abstract type AbstractDatasets end

import Distributions: rand, logpdf
export rand, logpdf

using Random: AbstractRNG, MersenneTwister
using Distributions: ContinuousMultivariateDistribution, MixtureModel, MvNormal

### Ring

struct Ring{T<:AbstractFloat} <: ContinuousMultivariateDistribution
    mixturemodel
end

function Ring(n_clusters::Int, s::T, σ::T) where {T<:AbstractFloat}
    π_typed = convert(T, π)
    cluster_indices = collect(0:n_clusters-1)
    base_angle = π_typed * 2 / n_clusters
    angle = (base_angle .* cluster_indices) .- π_typed / 2
    μ = [s * cos.(angle) s * sin.(angle)]'
    return Ring{T}(MixtureModel([MvNormal(μ[:,i], σ) for i in 1:size(μ, 2)]))
end

rand(rng::AbstractRNG, dataset::Ring{T}, n::Int) where {T} = convert.(T, rand(rng, dataset.mixturemodel, n))

logpdf(dataset::Ring, x::AbstractArray{<:AbstractFloat,2}) = logpdf(dataset.mixturemodel, x)

struct RingDataset <: AbstractDatasets
    X
    ring
end

Tz(theta) = [cos(theta) 0 sin(theta); 0 1 0; -sin(theta) 0 cos(theta)]

function RingDataset(
    n_data::Int,
    n_mixtures::Int,
    distance::T,
    var::T,
    z_angle::Union{Nothing, T}=nothing, 
    z_noise_level::T=1f-1; 
    seed::Int=1
) where {T<:AbstractFloat}
    rng = MersenneTwister(seed)
    ring = Ring(n_mixtures, distance, var)
    X = rand(rng, ring, n_data)
    if !isnothing(z_angle)
        X = Tz(z_angle) * cat(X, randn(T, 1, n_data) * z_noise_level; dims=1)
    end
    return RingDataset(X, ring)
end

export RingDataset

### Feature

struct FeatureDataset
    X
    features
end

function FeatureDataset(n_data::Int, features::Matrix{T}; seed::Int=1) where {T}
    rng = MersenneTwister(seed)
    n_features = size(dataset.features, 2)
    activation_matrix = rand(rng, n_features, n_data) .> 0.5
    X = features * activation_matrix
    return FeatureDataset(X, features)
end

include("features.jl")  # actual features

export FeatureDataset, get_features_griffiths2011, get_features_xu2019

export Datasets

end # module
