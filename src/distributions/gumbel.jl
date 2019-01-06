abstract type AbstractGumbelSoftmax{T} end

"""
The Gumbel-Softmax distributions.

NOTE: parameters are in batch.

Ref: https://arxiv.org/abs/1611.01144
"""
struct GumbelSoftmax{T} <: AbstractGumbelSoftmax{T}
    p::T
end

struct GumbelSoftmax2D{T} <: AbstractGumbelSoftmax{T}
    p::T
    function GumbelSoftmax2D{T}(p) where {T}
        FT = eltype(p)
        new(hcat(p, one(FT) .- p)')
    end
end

"""
    rand(gs::GumbelSoftmax{T}; τ=0.1)

Sample from the Gumbel-Softmax distributions.
"""
function rand(gs::AbstractGumbelSoftmax{AT}; τ=0.1) where {AT}
    FT = eltype(gs.p)

    u = rand(FT, size(gs.p)...)
    g = AT(-log.(-log.(u)))

    logit = g .+ log.(gs.p)
    exp_logit = exp.(logit ./ τ)
    return exp_logit ./ sum(exp_logit; dims=1)
end

"""
The Gumbel-Bernoulli distributions.

NOTE: parameters are in batch.

Ref: https://arxiv.org/abs/1611.01144
"""
struct GumbelBernoulli{T}
    p::T
end

"""
    rand(gb::GumbelBernoulli{AT}; τ=0.1) where {AT}

Sample from Gumbel-Bernoulli distributions.
"""
function rand(gb::GumbelBernoulli{AT}; τ=0.1) where {AT}
    # TODO: re-implement this `rand` using the same procedure for `GumbelBernoulliLogit`
    FT = eltype(gb.p)
    sz = size(gb.p)

    u0 = rand(FT, sz...); g0 = AT(-log.(-log.(u0)))
    u1 = rand(FT, sz...); g1 = AT(-log.(-log.(u1)))

    logit0 = (g0 .+ log.(one(FT) .- gb.p)) ./ τ
    logit1 = (g1 .+ log.(gb.p)) ./ τ

    logit_max = max.(logit0, logit1)
    logit1_minus_max = logit1 .- logit_max

    logx = logit1_minus_max .- log.(exp.(logit0 - logit_max) .+ exp.(logit1_minus_max))
    return exp.(logx)
end

"""
The Gumbel-Bernoulli distributions in logit space.

NOTE: parameters are in batch.

Ref: https://arxiv.org/abs/1611.01144
"""
struct GumbelBernoulliLogit{T}
    logitp::T
end

"""
    sample_logit_from_bernoulli(lp; τ=FT(0.1))

Sample logit from Bernoulli distributions by logit.

NOTE: `lp` is assumed to be in batch

Ref: https://arxiv.org/abs/1611.00712
"""
function rand(gbl::GumbelBernoulliLogit{AT}; τ=0.1) where {AT}
    FT = eltype(gbl.logitp)

    u = AT(rand(FT, size(gbl.logitp)...))

    logit = log.(u) - log.(one(FT) .- u)

    logitx = (gbl.logitp + logit) ./ τ
    return logitx
end

"""
    logpdf(gbl::GumbelBernoulliLogit, logitx; τ=0.1)

Compute ``GumbelBernoulli(logitx; logitp)``.

NOTE: `logitp` and `logitx` are assumed to be in batch.

Ref: https://arxiv.org/abs/1611.00712
"""
function logpdf(gbl::GumbelBernoulliLogit, logitx; τ=0.1)
    # WARNING: this function is not tested.
    exp_term = gbl.logitp .- logitx .* τ
    lp = exp_term .+ log(τ) .- FT(2.0) .* softplus.(exp_term)
    return lp
end
