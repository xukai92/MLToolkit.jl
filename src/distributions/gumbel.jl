abstract type AbstractBatchGumbelSoftmax{T} end

"""
The Gumbel-Softmax distributions.

NOTE: parameters are in batch.

Ref: https://arxiv.org/abs/1611.01144
"""
struct BatchGumbelSoftmax{T} <: AbstractBatchGumbelSoftmax{T}
    p::T
    τ::FT
end

BatchGumbelSoftmax(p; τ=FT(0.2)) = BatchGumbelSoftmax(p, τ)

struct BatchGumbelSoftmax2D{T} <: AbstractBatchGumbelSoftmax{T}
    p::T
    τ::FT
end

BatchGumbelSoftmax2D(p::T; τ=FT(0.2)) where {T} = BatchGumbelSoftmax2D(T(hcat(p, one(eltype(p)) .- p)'), τ)

function _u2gumbel(u)
    _eps = eps(eltype(u))
    return -log.(-log.(u .+ _eps) .+ _eps)
end

_u2gumbelback(u, g) = one(eltype(u)) ./ exp.(-g) ./ (u .+ eps(eltype(u)))

Knet.@primitive _u2gumbel(u),dy,g  dy.*_u2gumbelback(u,g)

function _g2softmax(g, p, τ)
    logit = g .+ log.(p .+ eps(eltype(g)))
    return Knet.softmax(logit ./ τ; dims=1)
end

"""
    rand(gs::AbstractBatchGumbelSoftmax)

Sample from the Gumbel-Softmax distributions.
"""
function rand(gs::AbstractBatchGumbelSoftmax{T}) where {T}
    u = randsimilar(gs.p)
    g = _u2gumbel(u)
    return _g2softmax(g, gs.p, gs.τ)
end

"""
    rand(gs::AbstractBatchGumbelSoftmax, n::Int)

Generate multiple samples from the Gumbel-Softmax distributions.
"""
function rand(gs::AbstractBatchGumbelSoftmax{T}, n::Int) where {T}
    @assert size(gs.p, 2) == 1
    u = randsimilar(gs.p, n)
    g = _u2gumbel(u)
    return _g2softmax(g, gs.p, gs.τ)
end

mean(gs::AbstractBatchGumbelSoftmax) = gs.p

"""
The Gumbel-Bernoulli distributions.

NOTE: parameters are in batch.

Ref: https://arxiv.org/abs/1611.01144
"""
struct BatchGumbelBernoulli{T}
    p::T
    τ::FT
end

BatchGumbelBernoulli(p; τ=FT(0.2)) = BatchGumbelBernoulli(p, τ)

"""
    rand(gb::BatchGumbelBernoulli)

Sample from Gumbel-Bernoulli distributions.
"""
function rand(gb::BatchGumbelBernoulli{T}) where {T}
    # TODO: re-implement this `rand` using the same procedure for `BatchGumbelBernoulliLogit`
    _eps = eps(eltype(gb.p))
    _one = one(eltype(gb.p))
    τ = gb.τ

    u0 = randsimilar(gb.p); g0 = _u2gumbel(u0)
    u1 = randsimilar(gb.p); g1 = _u2gumbel(u1)

    logit0 = (g0 .+ log.(_one + _eps .- gb.p)) ./ τ
    logit1 = (g1 .+ log.(gb.p .+ _eps)) ./ τ

    logit_max = max.(logit0, logit1)
    logit1_minus_max = logit1 .- logit_max

    logx = logit1_minus_max .- log.(exp.(logit0 - logit_max) .+ exp.(logit1_minus_max))
    return exp.(logx)
end

"""
    logpdf(bgb::BatchGumbelBernoulli, x)

Compute ``GumbelBernoulli(x; p)``.

NOTE: `p` and `x` are assumed to be in batch.

Ref: https://arxiv.org/abs/1611.01144
"""
function logpdf(bgb::BatchGumbelBernoulli, x)
    _eps = eps(FT)
    τ = bgb.τ
    α = bgb.p ./ (1 .- bgb.p .+ _eps)
    xstabe = x .+ _eps
    omxstabe = 1 .- x .+ _eps
    return log(τ) .+ log.(α) + (-τ - 1) * (log.(xstabe) + log.(omxstabe)) - 2 * (log.(α .* xstabe.^(-τ) + omxstabe.^(-τ) .+ _eps))
end

"""
    logpdfcov(bgb::BatchGumbelBernoulli, x)

Compute ``GumbelBernoulli(x; p)`` by using that of `GumbelBernoulli(logitx; logitp)`
together with change of variable (CoV) rules.

NOTE: `p` and `x` are assumed to be in batch.
"""
function logpdfCoV(bgb::BatchGumbelBernoulli, x)
    _eps = eps(FT)
    _1m2eps = 1 - 2 * _eps
    logitp = logit.(bgb.p .* _1m2eps .+ _eps)
    logitx = logit.(x .* _1m2eps .+ _eps)
    lp = logpdflogit(BatchGumbelBernoulliLogit(logitp; τ=bgb.τ), logitx)
    _eps = eps(FT)
    return lp - log.(x .* (1 .- x) .+ _eps)
end

mean(gb::BatchGumbelBernoulli) = gb.p

"""
The Gumbel-Bernoulli distributions in logit space.

NOTE: parameters are in batch.

Ref: https://arxiv.org/abs/1611.01144
"""
struct BatchGumbelBernoulliLogit{T}
    logitp::T
    τ::FT
end

BatchGumbelBernoulliLogit(logitp; τ=FT(0.2)) = BatchGumbelBernoulliLogit(logitp, τ)

"""
    logitrand(gbl::BatchGumbelBernoulliLogit{T}; τ=gbl.τ) where {T}

Sample logit from Bernoulli distributions by logit.

NOTE: `lp` is assumed to be in batch

Ref: https://arxiv.org/abs/1611.00712
"""
function logitrand(gbl::BatchGumbelBernoulliLogit{T}; τ=gbl.τ) where {T}
    FT = eltype(gbl.logitp)
    _eps = eps(FT)
    _one = one(FT)

    u = randsimilar(gbl.logitp)

    logit = log.(u .+ _eps) - log.(_one + _eps .- u)

    logitx = (gbl.logitp + logit) ./ τ
    return logitx
end

"""
    logpdflogit(gbl::BatchGumbelBernoulliLogit, logitx)

Compute `GumbelBernoulli(logitx; logitp)`.

NOTE: `logitp` and `logitx` are assumed to be in batch.

WARN: this function is not tested.

Ref: https://arxiv.org/abs/1611.00712
"""
function logpdflogit(gbl::BatchGumbelBernoulliLogit, logitx; τ=gbl.τ)
    exp_term = gbl.logitp .- logitx .* τ
    lp = exp_term .+ log(τ) .- FT(2.0) .* softplus.(exp_term)
    return lp
end

mean(gbl::BatchGumbelBernoulliLogit) = Knet.sigm.(gbl.logitp)
