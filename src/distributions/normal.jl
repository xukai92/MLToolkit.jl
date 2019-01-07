using LinearAlgebra: det, tr, inv
using Distributions: MvNormal

"""
Normal distribution with parameters (possibly) in batch.
"""
struct BatchNormal{T}
    μ::T    # mean
    Σ::T    # variance
end

"""
    rand(dn::BatchNormal)

Sample from the Normal distribution.

NOTE: `dn.μ` and `dn.Σ` are assumed to be in batch.

Ref: https://arxiv.org/pdf/1312.6114.pdf
"""
function rand(dn::BatchNormal)
    ϵ = AT(randn(eltype(dn.μ), size(dn.μ)...))
    return dn.μ + sqrt.(dn.Σ) .* ϵ
end

"""
    logpdf(dn::BatchNormal, x)

Compute ``Normal(x; μ, Σ)`` in an element-wise manner.

NOTE: `n.μ`, `n.Σ` and `x` are assumed to be in batch.
"""
function logpdf(dn::BatchNormal, x)
    FT = eltype(dn.μ)
    diff = x .- dn.μ
    return -FT(0.5) .* (log(2 * FT(pi)) .+ log.(dn.Σ) .+ diff .* diff ./ dn.Σ)
end

"""
    kl(dn1::BatchNormal, dn2::BatchNormal)

Compute ``KL(Normal_1||Normal_2)``.
"""
function kl(dn1::BatchNormal, dn2::BatchNormal)
    FT = eltype(dn1.μ)
    if eltype(dn2.μ) != FT
        @warn "FT are different for bn1 and bn2" eltype(bn1.μ) eltype(bn2.μ)
    end
    diff = dn2.μ .- dn1.μ
    Σ1 = dn1.Σ
    Σ2 = dn2.Σ
    return FT(0.5) .* (log.(Σ2) .- log.(Σ1) .- 1 .+ Σ1 ./ Σ2 .+ diff .* diff ./ Σ2)
end

"""
    kl(mvn1::MvNormal, mvn2::MvNormal)

Compute ``KL(MvNormal_1||MvNormal_2)``.
"""
function kl(mvn1::MvNormal, mvn2::MvNormal)
    FT = eltype(mvn1.μ)
    d = length(mvn1.μ)
    diff = mvn2.μ .- mvn1.μ
    Σ1 = Matrix(mvn1.Σ)
    Σ2 = Matrix(mvn2.Σ)
    return FT(0.5) * (log(det(Σ2)) - log(det(Σ1)) - d + tr(inv(Σ2) * Σ1) + diff' * inv(Σ2) * diff)
end
