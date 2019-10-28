### Uniform

struct DiagUniform{T} <: ContinuousMultivariateDistribution
    D::Int
    L::T
    U::T
end

function rand(rng::AbstractRNG, d::DiagUniform, n::Int)
    s = (d.U - d.L)
    t = s / 2
    return s * rand(rng, FT[], d.D, n) .- t
end

function logpdf(d::DiagUniform, x::AbstractVecOrMat)
    size(x, 1) != d.D && throw(DimensionMismatch())
    return _logpdf(d, x)
end

_logpdf(d::DiagUniform, x::AbstractVector{T}) where {T<:AbstractFloat} = one(T) / ((d.U - d.L) * d.D)
_logpdf(d::DiagUniform, x::AbstractMatrix{T}) where {T<:AbstractFloat} = one(T) / ((d.U - d.L) * d.D) * ones(size(x, 2))

DiagUniform(D::Int) = DiagUniform(D, -1, 1)

### Std Normal

struct DiagStdNormal <: ContinuousMultivariateDistribution
    D::Int
end

rand(rng::AbstractRNG, d::DiagStdNormal, n::Int) = randn(rng, FT[], d.D, n)

_logpdf(d::DiagStdNormal, x::AbstractVector{T}) where {T<:AbstractFloat} = -(d.D * log(2T(π)) + sum(abs2, x)) / 2
_logpdf(d::DiagStdNormal, x::AbstractMatrix{T}) where {T<:AbstractFloat} = -(d.D * log(2T(π)) .+ sum(abs2, x; dims=1)') / 2
