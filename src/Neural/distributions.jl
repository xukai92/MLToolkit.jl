import Random, Distributions

### Uniform

struct DiagUniform{T} <: Distributions.ContinuousMultivariateDistribution
    D::Int
    L::T
    U::T
end

function Distributions.rand(rng::AbstractRNG, d::DiagUniform, n::Int)
    s = (d.U - d.L)
    t = s / 2
    return s * rand(rng, FT[], b.D, n) .- t
end

function Distributions.logpdf(d::DiagUniform, x::AbstractVecOrMat)
    size(x, 1) != d.D && throw(DimensionMismatch())
    return _logpdf(d, x)
end

_logpdf(d::DiagUniform, x::AbstractVector{T}) where {T<:AbstractFloat} = one(T) / ((d.U - d.L) * d.D)
_logpdf(d::DiagUniform, x::AbstractMatrix{T}) where {T<:AbstractFloat} = one(T) / ((d.U - d.L) * d.D) * ones(size(x, 2))

DiagUniform(D::Int) = DiagUniform(D, -1, 1)

### Normal

struct DiagStdNormal <: Distributions.ContinuousMultivariateDistribution
    D::Int
end

Distributions.rand(rng::AbstractRNG, b::DiagStdNormal, n::Int) = randn(rng, FT[], b.D, n)

_logpdf(d::DiagStdNormal, x::AbstractVector{T}) where {T<:AbstractFloat} = -(d.D * log(2T(π)) + sum(abs2, x)) / 2
_logpdf(d::DiagStdNormal, x::AbstractMatrix{T}) where {T<:AbstractFloat} = -(d.D * log(2T(π)) .+ sum(abs2, x; dims=1)') / 2

DiagNormal(D::Int) = DiagStdNormal(D)
