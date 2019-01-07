using Distributions: Dirichlet
using SpecialFunctions: lgamma, digamma, lbeta, beta
using Base.MathConstants: γ

struct BatchKumaraswamy{T}
    a::T
    b::T
end

function _u2kumaraswamysample(T, u, kuma::BatchKumaraswamy)
    _one = one(T)
    return (_one .- u.^(_one ./ kuma.b)).^(_one ./ kuma.a)
end

function _u2logkumaraswamysample(T, u, kuma::BatchKumaraswamy)
    _one = one(T)
    _eps = eps(T)
    return log.(_one + _eps .- exp.(log.(u) ./ (kuma.b .+ _eps))) ./ (kuma.a .+ _eps)
end

"""
    rand(kuma::BatchKumaraswamy{AT}) where {AT}

Sample from Kumaraswamy distribution.

NOTE: `k.a` and `k.b` are assumed to be in batch

Ref: https://arxiv.org/abs/1605.06197
"""
function rand(kuma::BatchKumaraswamy{AT}) where {AT}
    u = AT(rand(FT, size(kuma.a)...))
    x = _u2kumaraswamysample(eltype(kuma.a), u, kuma)
    return x
end

function rand(kuma::BatchKumaraswamy{T}, dims::Integer...) where {T<:Real}
    u = rand(FT, dims...)
    x = _u2kumaraswamysample(T, u, kuma)
    return x
end

function logrand(kuma::BatchKumaraswamy{AT}) where {AT}
    u = AT(rand(FT, size(kuma.a)...))
    logx = _u2logkumaraswamysample(eltype(kuma.a), u, kuma)
    return logx
end

function logrand(kuma::BatchKumaraswamy{T}, dims::Integer...) where {T<:Real}
    u = rand(FT, dims...)
    logx = _u2logkumaraswamysample(T, u, kuma)
    return logx
end

"""
    logpdf(kuma::BatchKumaraswamy, x)

Compute ``Kumaraswamy(x; a, b)``.
"""
function logpdf(kuma::BatchKumaraswamy, x)
    # WARNING: this function is not tested.
    _one = one(FT)
    _eps = eps(FT)
    lp = log(kuma.a) .+
         log(kuma.b) .+
         (kuma.a .- _one) .* log.(x .+ _eps) .+
         (kuma.b .- _one) .* log.(_one .- x.^kuma.a)
    return lp
end

"""
    kl(d1::Dirichlet, d2::Dirichlet)

Compute ``KL(Dir(α)||Dir(β))`` where ``α = [α_1, \\dots, α_K]`` and ``β = [β_1, \\dots, β_K]``.

Ref: http://bariskurt.com/kullback-leibler-divergence-between-two-dirichlet-and-beta-distributions/
"""
function kl(d1::Dirichlet, d2::Dirichlet)
    α = d1.alpha
    β = d2.alpha

    α0 = sum(α)
    β0 = sum(β)

    kl = (lgamma.(α0) .- sum(lgamma.(α))) .-
         (lgamma(β0) - sum(lgamma.(β))) .+
         sum((α .- β) .* (digamma.(α) .- digamma(α0)))
    return kl
end

struct BatchBeta{T}
    α::T
    β::T
end

"""
    kl(bb1::BatchBeta, bb2::BatchBeta)

Compute ``KL(Beta(α1, β1)||Beta(α2, β2))``.
"""
function kl(bb1::BatchBeta, bb2::BatchBeta)
    α1 = bb1.α
    β1 = bb1.β
    α2 = bb2.α
    β2 = bb2.β

    α0 = α1 .+ β1
    β0 = α2 .+ β2

    kl = (lgamma.(α0) .- (lgamma.(α1) .+ lgamma.(β1))) .-
         (lgamma.(β0) .- (lgamma.(α2) .+ lgamma.(β2))) .+
         (α1 - α2) .* (digamma.(α1) - digamma.(α0)) .+
         (β1 - β2) .* (digamma.(β1) - digamma.(α0))
    return kl
end

"""
    kl(kuma::BatchKumaraswamy, bb::BatchBeta; M=10)

Compute ``KL(Kumaraswamy(a, b)||Beta(α, β))``.

NOTE: only `a` and `b` are assumed to be in batch

NOTE: this function is not tested
"""
function kl(kuma::BatchKumaraswamy, bb::BatchBeta; M=10)
    a = kuma.a
    b = kuma.b
    α = bb.α
    β = bb.β
    FT = eltype(a)
    _one = one(FT)

    @assert M > 0
    acc = _one ./ (_one .+ a .* b) .* beta.(_one ./ a, b)
    for m = 2:M
        acc = acc .+ _one ./ (FT(m) .+ a .* b) .* beta.(FT(m) ./ a, b)
    end

    kl = (a .- α) ./ a .* (-FT(γ) .- digamma.(b) .- _one ./ b) .+ log.(a .* b) .+
         lbeta(α, β) .- (b .- _one) ./ b .+ (β - _one) .* b .* acc
    return kl
end
