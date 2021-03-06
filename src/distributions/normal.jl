using LinearAlgebra: det, tr, inv
using Distributions: Normal, MvNormal

"""
Normal distribution with parameters (possibly) in batch.
"""
struct BatchNormal{T}
    μ::T    # mean
    Σ       # variance
end

"""
    rand(dn::BatchNormal)

Sample from the Normal distribution.

NOTE: `dn.μ` and `dn.Σ` are assumed to be in batch.

Ref: https://arxiv.org/pdf/1312.6114.pdf
"""
function rand(dn::BatchNormal{T}) where {T}
    ϵ = randnsimilar(dn.μ)
    return dn.μ + sqrt.(dn.Σ) .* ϵ
end

function rand(dn::BatchNormal{T}, dims...) where {T<:Real}
    return rand(Normal(dn.μ, sqrt(dn.Σ)), dims...)
end

"""
    logpdf(dn::BatchNormal, x)

Compute ``Normal(x; μ, Σ)`` in an element-wise manner.

NOTE: `n.μ`, `n.Σ` and `x` are assumed to be in batch.
"""
function logpdf(dn::BatchNormal, x)
    diff = x .- dn.μ
    return -(log(2 * eltype(dn.μ)(pi)) .+ log.(dn.Σ) .+ diff .* diff ./ dn.Σ) ./ 2
end

mean(bn::BatchNormal) = bn.μ
mode(bn::BatchNormal) = bn.μ

struct BatchNormalLogVar{T}
    μ::T    # mean
    logΣ    # log-variance
end

function rand(dn::BatchNormalLogVar{T}) where {T}
    ϵ = randnsimilar(dn.μ)
    return dn.μ + exp.(dn.logΣ ./ 2) .* ϵ
end

function logpdf(dn::BatchNormalLogVar, x)
    diff = x .- dn.μ
    return -(log(2 * eltype(dn.μ)(pi)) .+ dn.logΣ .+ diff .* diff ./ exp.(dn.logΣ)) ./ 2
end

mean(bn::BatchNormalLogVar) = bn.μ
mode(bn::BatchNormalLogVar) = bn.μ

"""
    kldiv(bn1::BatchNormal, bn2::BatchNormal)

Compute ``KL(Normal_1||Normal_2)``.
"""
function kldiv(bn1::BatchNormal, bn2::BatchNormal)
    if eltype(bn1.μ) != eltype(bn2.μ)
        @warn "Float type are different for bn1 and bn2" eltype(bn1.μ) eltype(bn2.μ)
    end
    diff = bn2.μ .- bn1.μ
    Σ1 = bn1.Σ
    Σ2 = bn2.Σ
    return (log.(Σ2) .- log.(Σ1) .- 1 .+ Σ1 ./ Σ2 .+ diff .* diff ./ Σ2) / 2
end

function kldiv(bnlv1::BatchNormalLogVar, bn2::BatchNormal)
    if eltype(bnlv1.μ) != eltype(bn2.μ)
        @warn "Float type are different for bnlv1 and bn2" eltype(bnlv1.μ) eltype(bn2.μ)
    end
    diff = bn2.μ .- bnlv1.μ
    logΣ1 = bnlv1.logΣ
    Σ2 = bn2.Σ
    return (log.(Σ2) .- logΣ1 .- 1 .+ exp.(logΣ1) ./ Σ2 .+ diff .* diff ./ Σ2) ./ 2
end

"""
    kldiv(mvn1::MvNormal, mvn2::MvNormal)

Compute ``KL(MvNormal_1||MvNormal_2)``.
"""
function kldiv(mvn1::MvNormal, mvn2::MvNormal)
    d = length(mvn1.μ)
    diff = mvn2.μ .- mvn1.μ
    Σ1 = Matrix(mvn1.Σ)
    Σ2 = Matrix(mvn2.Σ)
    return (log(det(Σ2)) - log(det(Σ1)) - d + tr(inv(Σ2) * Σ1) + diff' * inv(Σ2) * diff) ./ 2
end
