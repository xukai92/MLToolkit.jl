using Random: AbstractRNG
using Distributions: Distributions, rand, logpdf, ContinuousMultivariateDistribution, MixtureModel, MvNormal

struct Ring{T<:AbstractFloat} <: ContinuousMultivariateDistribution
    mixturemodel
end

function Ring(n_clusters::Int, distance::T, var::T) where {T<:AbstractFloat}
    pi_typed = convert(T, pi)
    cluster_idcs = collect(0:n_clusters-1)
    base_angle = pi_typed * 2 / n_clusters
    angle = (base_angle .* cluster_idcs) .- pi_typed / 2
    mean = [distance * cos.(angle) distance * sin.(angle)]'
    return Ring{T}(MixtureModel([MvNormal(mean[:,i], var) for i in 1:n_clusters]))
end

Distributions.rand(rng::AbstractRNG, ring::Ring{T}, n::Int) where {T<:AbstractFloat} = convert.(T, rand(rng, ring.mixturemodel, n))

Distributions.logpdf(dataset::Ring, X::AbstractArray{<:AbstractFloat,2}) = logpdf(dataset.mixturemodel, X)

abstract type RingDataset <: AbstractDataset end

n_display(d::RingDataset) = 2_000

struct TwoDimRingDataset <: RingDataset
    X
end

function RingDataset(
    n_data::Int,
    n_clusters::Int=10,
    distance::T=2.5f0,
    var::T=0.25f0; 
    seed::Int=1
) where {T<:AbstractFloat}
    rng = MersenneTwister(seed)
    ring = Ring(n_clusters, distance, var)
    X = rand(rng, ring, n_data)
    return TwoDimRingDataset(X)
end

struct ThreeDimRingDataset <: RingDataset
    X
end

Tz(theta) = [cos(theta) 0 sin(theta); 0 1 0; -sin(theta) 0 cos(theta)]

function RingDataset(
    n_data::Int,
    n_clusters::Int,
    distance::T,
    var::T,
    z_angle::T, 
    z_noise_level::T;
    seed::Int=1
) where {T<:AbstractFloat}
    rng = MersenneTwister(seed)
    ring = Ring(n_clusters, distance, var)
    X = rand(rng, ring, n_data)
    X = Tz(z_angle) * cat(X, randn(T, 1, n_data) * z_noise_level; dims=1)
    return ThreeDimRingDataset(X)
end
