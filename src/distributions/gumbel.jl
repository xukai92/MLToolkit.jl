abstract type AbstractBatchGumbelSoftmax{T} end

"""
The Gumbel-Softmax distributions.

NOTE: parameters are in batch.

Ref: https://arxiv.org/abs/1611.01144
"""
struct BatchGumbelSoftmax{T} <: AbstractBatchGumbelSoftmax{T}
    p::T
end

struct BatchGumbelSoftmax2D{T} <: AbstractBatchGumbelSoftmax{T}
    p::T
    function BatchGumbelSoftmax2D{T}(p) where {T}
        FT = eltype(p)
        new(hcat(p, one(FT) .- p)')
    end
end

function _u2gumbel(T, u)
    _eps = eps(T)
    return -log.(-log.(u .+ _eps) .+ _eps)
end

"""
    rand(gs::AbstractBatchGumbelSoftmax; τ=FT(0.2))

Sample from the Gumbel-Softmax distributions.
"""
function rand(gs::AbstractBatchGumbelSoftmax; τ=FT(0.2))
    FT = eltype(gs.p)
    _eps = eps(FT)

    u = rand(FT, size(gs.p)...)
    g = AT(_u2gumbel(FT, u))

    logit = g .+ log.(gs.p .+ _eps)
    exp_logit = exp.(logit ./ τ)
    return exp_logit ./ sum(exp_logit; dims=1)
end

mean(gs::AbstractBatchGumbelSoftmax) = gs.p

"""
The Gumbel-Bernoulli distributions.

NOTE: parameters are in batch.

Ref: https://arxiv.org/abs/1611.01144
"""
struct BatchGumbelBernoulli{T}
    p::T
end

"""
    rand(gb::BatchGumbelBernoulli; τ=FT(0.2))

Sample from Gumbel-Bernoulli distributions.
"""
function rand(gb::BatchGumbelBernoulli; τ=FT(0.2))
    # TODO: re-implement this `rand` using the same procedure for `BatchGumbelBernoulliLogit`
    FT = eltype(gb.p)
    sz = size(gb.p)
    _eps = eps(FT)
    _one = one(FT)

    u0 = rand(FT, sz...); g0 = AT(_u2gumbel(FT, u0))
    u1 = rand(FT, sz...); g1 = AT(_u2gumbel(FT, u1))

    logit0 = (g0 .+ log.(_one + _eps .- gb.p)) ./ τ
    logit1 = (g1 .+ log.(gb.p .+ _eps)) ./ τ

    logit_max = max.(logit0, logit1)
    logit1_minus_max = logit1 .- logit_max

    logx = logit1_minus_max .- log.(exp.(logit0 - logit_max) .+ exp.(logit1_minus_max))
    return exp.(logx)
end

mean(gb::BatchGumbelBernoulli) = gb.p

"""
The Gumbel-Bernoulli distributions in logit space.

NOTE: parameters are in batch.

Ref: https://arxiv.org/abs/1611.01144
"""
struct BatchGumbelBernoulliLogit{T}
    logitp::T
end

"""
    logitrand(gbl::BatchGumbelBernoulliLogit; τ=FT(0.2))

Sample logit from Bernoulli distributions by logit.

NOTE: `lp` is assumed to be in batch

Ref: https://arxiv.org/abs/1611.00712
"""
function logitrand(gbl::BatchGumbelBernoulliLogit; τ=FT(0.2))
    FT = eltype(gbl.logitp)
    _eps = eps(FT)
    _one = one(FT)

    u = AT(rand(FT, size(gbl.logitp)...))

    logit = log.(u .+ _eps) - log.(_one + _eps .- u)

    logitx = (gbl.logitp + logit) ./ τ
    return logitx
end

"""
    logpdflogit(gbl::BatchGumbelBernoulliLogit, logitx; τ=FT(0.2))

Compute ``GumbelBernoulli(logitx; logitp)``.

NOTE: `logitp` and `logitx` are assumed to be in batch.

WARN: this function is not tested.

Ref: https://arxiv.org/abs/1611.00712
"""
function logpdflogit(gbl::BatchGumbelBernoulliLogit, logitx; τ=FT(0.2))
    exp_term = gbl.logitp .- logitx .* τ
    lp = exp_term .+ log(τ) .- FT(2.0) .* softplus.(exp_term)
    return lp
end

mean(gbl::BatchGumbelBernoulliLogit) = Knet.sigm.(gbl.logitp)
