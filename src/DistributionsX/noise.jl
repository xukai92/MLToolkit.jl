### Uniform

struct UniformNoise <: ContinuousMultivariateDistribution
    D::Int
end

function rand(rng::AbstractRNG, d::UniformNoise, n::Int)
    return 2 * rand(rng, FloatT[], d.D, n) .- 1
end

function logpdf(d::UniformNoise, x::AbstractVecOrMat)
    size(x, 1) != d.D && throw(DimensionMismatch())
    return _logpdf(d, x)
end

_logpdf(d::UniformNoise, x::AbstractVector{T}) where {T<:AbstractFloat} = one(T) / 2d.D
_logpdf(d::UniformNoise, x::AbstractMatrix{T}) where {T<:AbstractFloat} = one(T) / 2d.D * ones(size(x, 2))

### Std Normal

struct GaussianNoise <: ContinuousMultivariateDistribution
    D::Int
end

rand(rng::AbstractRNG, d::GaussianNoise, n::Int) = randn(rng, FloatT[], d.D, n)

_constant(d::GaussianNoise, x) = d.D * 2eltype(x)(π)
_logpdf(d::GaussianNoise, x::AbstractVector) = -(_constant(d, x) + sum(abs2, x)) / 2
_logpdf(d::GaussianNoise, x::AbstractMatrix) = -(_constant(d, x) .+ sum(abs2, x; dims=1)') / 2
