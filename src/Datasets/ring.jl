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

struct RingDataset{D} <: AbstractDataset{D}
    X
    Xt
end

n_display(d::RingDataset) = 2_000

Tz(angle) = [cos(angle) 0 sin(angle); 0 1 0; -sin(angle) 0 cos(angle)]

function RingDataset(
    n_data::Int,
    n_clusters::Int=10,
    distance::T1=2.5f0,
    var::T1=0.25f0,
    z_angle::T2=nothing,
    z_std::T2=nothing;
    seed::Int=1,
    test_ratio=1/6,
    n_test::Int=ratio2num(n_data, test_ratio),
) where {T1<:AbstractFloat, T2<:Union{T1, Nothing}}
    rng = MersenneTwister(seed)
    ring = Ring(n_clusters, distance, var)
    function make3d(X)
        return Tz(z_angle) * cat(X, randn(T1, 1, size(X, 2)) * z_std; dims=1)
    end
    X = rand(rng, ring, n_data)
    Xt = rand(rng, ring, n_test)
    if isnothing(z_angle) || isnothing(z_std)
        dataset = RingDataset{2}(X, Xt)
    else
        dataset = RingDataset{3}(make3d(X), make3d(Xt))
    end
    return dataset
end
