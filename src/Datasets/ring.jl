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

Distributions.rand(rng::AbstractRNG, ring::Ring{T}, n::Int) where {T<:AbstractFloat} = 
    convert.(T, rand(rng, ring.mixturemodel, n))
Distributions.logpdf(dataset::Ring, X::AbstractArray{<:AbstractFloat,2}) = logpdf(dataset.mixturemodel, X)

struct RingDataset{D} <: AbstractDataset{D}
    X
    Xt
end

n_display(d::RingDataset) = 2_000

function RingDataset(
    n_data::Int,
    z_angle::T1=Float32(pi / 3),
    z_std::T1=1f-1,
    n_clusters::Int=10,
    distance::T2=25f-1,
    var::T2=25f-2;
    seed::Int=1,
    test_ratio=1/6,
    n_test::Int=ratio2num(n_data, test_ratio),
    rng=MersenneTwister(seed),
) where {T<:AbstractFloat, T1<:Union{Nothing, T}, T2<:T}
    ring = Ring(n_clusters, distance, var)
    Tz(angle) = [cos(angle) 0 sin(angle); 0 1 0; -sin(angle) 0 cos(angle)]
    make3d(X) = Tz(z_angle) * cat(X, randn(T1, 1, size(X, 2)) * z_std; dims=1)
    X, Xt = rand(rng, ring, n_data), rand(rng, ring, n_test)
    if isnothing(z_angle) || isnothing(z_std)
        @info "Oh you just get the 2D ring dataset" n_data=n_data n_test=n_test n_clusters=n_clusters distance=distance var=var
        return RingDataset{2}(X, Xt)
    else
        X, Xt = make3d.((X, Xt))
        @info "Oh you just get the 3D ring dataset" n_data=n_data n_test=n_test n_clusters=n_clusters distance=distance var=var z_angle=z_angle z_std=z_std
        return RingDataset{3}(X, Xt)
    end
end

RingDataset{2}(n_data::Int=60_000, args...; kwargs...) = RingDataset(n_data, nothing, nothing, args...; kwargs...)
TwoDimRingDataset = RingDataset{2}

RingDataset{3}(n_data::Int=60_000, args...; kwargs...) = RingDataset(n_data, args...; kwargs...)
ThreeDimRingDataset = RingDataset{3}
