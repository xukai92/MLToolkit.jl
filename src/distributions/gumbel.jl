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

function _u2gumbel(T, u)
    _eps = eps(T)
    return -log.(-log.(u .+ _eps) .+ _eps)
end

"""
    rand(gs::AbstractBatchGumbelSoftmax)

Sample from the Gumbel-Softmax distributions.
"""
function rand(gs::AbstractBatchGumbelSoftmax{T}) where {T}
    FT = eltype(gs.p)
    _eps = eps(FT)

    u = rand(FT, size(gs.p)...)
    g = T(_u2gumbel(FT, u))

    logit = g .+ log.(gs.p .+ _eps)
    exp_logit = exp.(logit ./ gs.τ)
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
    τ::FT
end

BatchGumbelBernoulli(p; τ=FT(0.2)) = BatchGumbelBernoulli(p, τ)

"""
    rand(gb::BatchGumbelBernoulli)

Sample from Gumbel-Bernoulli distributions.
"""
function rand(gb::BatchGumbelBernoulli{T}) where {T}
    _T = isa(gb.p, AutoGrad.Result) ? typeof(value(gb)) : T

    # TODO: re-implement this `rand` using the same procedure for `BatchGumbelBernoulliLogit`
    FT = eltype(gb.p)
    sz = size(gb.p)
    _eps = eps(FT)
    _one = one(FT)
    τ = gb.τ

    u0 = rand(FT, sz...); g0 = _T(_u2gumbel(FT, u0))
    u1 = rand(FT, sz...); g1 = _T(_u2gumbel(FT, u1))

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
    logitrand(gbl::BatchGumbelBernoulliLogit)

Sample logit from Bernoulli distributions by logit.

NOTE: `lp` is assumed to be in batch

Ref: https://arxiv.org/abs/1611.00712
"""
function logitrand(gbl::BatchGumbelBernoulliLogit{T}) where {T}
    FT = eltype(gbl.logitp)
    _eps = eps(FT)
    _one = one(FT)
    τ = gbl.τ

    u = T(rand(FT, size(gbl.logitp)...))

    logit = log.(u .+ _eps) - log.(_one + _eps .- u)

    logitx = (gbl.logitp + logit) ./ τ
    return logitx
end

"""
    logpdflogit(gbl::BatchGumbelBernoulliLogit, logitx)

Compute `GumbelBernoulli(logitx; logitp)`.

NOTE: `logitp` and `logitx` are assumed to be in batch.
TODO: double check this function and make sure it obeys the change of variable

WARN: this function is not tested.

Ref: https://arxiv.org/abs/1611.00712
"""
function logpdflogit(gbl::BatchGumbelBernoulliLogit, logitx)
    τ = gbl.τ
    exp_term = gbl.logitp .- logitx .* τ
    lp = exp_term .+ log(τ) .- FT(2.0) .* softplus.(exp_term)
    return lp
end

mean(gbl::BatchGumbelBernoulliLogit) = Knet.sigm.(gbl.logitp)

# TODO: consider the design problem: should `τ` be a filed of gumbel distributions?
