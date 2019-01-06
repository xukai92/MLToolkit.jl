using LinearAlgebra: det, tr, inv

abstract type AbstractNormal end

struct UnivariateNormal{T<:Real} <: AbstractNormal
    μ::T    # mean
    Σ::T    # variance
    function UnivariateNormal{T}(μ, Σ) where {T<:Real}
        @assert Σ > 0 "Σ is not positve"
        return new{T}(μ, Σ)
    end
end
UnivariateNormal(μ::T, Σ::T) where {T<:Real} = UnivariateNormal{T}(μ, Σ)

"""
Diagonal Normal distribution with parameters in batch.
"""
struct DiagonalNormal{T} <: AbstractNormal
    μ::T    # mean
    Σ::T    # variance
end

struct DenseNormal <: AbstractNormal
    μ::AbstractVector   # mean
    Σ::AbstractMatrix   # covariance
end

"""
    rand(dn::DiagonalNormal{AT}) where AT

Sample from diagonal Normal distribution.

NOTE: `dn.μ` and `dn.Σ` are assumed to be in batch.

Ref: https://arxiv.org/pdf/1312.6114.pdf
"""
function rand(dn::DiagonalNormal{AT}) where AT
    ϵ = AT(randn(eltype(dn.μ), size(dn.μ)...))
    return dn.μ + sqrt.(dn.Σ) .* ϵ
end

"""
    logpdf(dn::DiagonalNormal, x)

Compute ``Normal(x; μ, Σ)`` in an element-wise manner.

NOTE: `n.μ`, `n.Σ` and `x` are assumed to be in batch.
"""
function logpdf(dn::DiagonalNormal, x)
    FT = eltype(dn.μ)
    d = size(dn.μ, 1)
    diff = x .- dn.μ
    return -FT(0.5) .* (d * log(2 * FT(pi)) .+ sum(log.(dn.Σ) .+ diff .* diff ./ dn.Σ; dims=1))
end

"""
    kl(dn1::DiagonalNormal, dn2::AbstractNormal)

Compute ``KL(Normal_1||Normal_2)``.
"""
function kl(dn1::DiagonalNormal, dn2::AbstractNormal)
    FT = eltype(dn1.μ)
    diff = dn2.μ .- dn1.μ
    return FT(0.5) .* sum(log.(dn2.Σ) .- log.(dn1.Σ) .- 1 .+ dn1.Σ ./ dn2.Σ .+ diff .* diff ./ dn2.Σ; dims=2)
end

"""
    kl(dn1::DenseNormal, dn2::DenseNormal)

Compute ``KL(Normal_1||Normal_2)``.
"""
function kl(dn1::DenseNormal, dn2::DenseNormal)
    FT = eltype(dn1.μ)
    d = length(dn1.μ)
    diff = dn2.μ .- dn1.μ
    return FT(0.5)* (log(det(dn2.Σ)) - log(det(dn1.Σ)) - d + tr(inv(dn2.Σ) * dn1.Σ) + diff' * inv(dn2.Σ) * diff)
end
