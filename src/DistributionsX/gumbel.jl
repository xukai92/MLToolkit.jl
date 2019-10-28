abstract type AbstractGumbelSoftmax{T} end

const τdefault = 0.2

"""
The Gumbel-Softmax distributions.

Ref: https://arxiv.org/abs/1611.01144
"""
struct GumbelSoftmax{T<:AbstractVecOrMat{<:AbstractFloat}} <: AbstractGumbelSoftmax{T}
    p::T
    τ
end

GumbelSoftmax(p; τ=eltype(p)(τdefault)) = GumbelSoftmax(p, τ)

function u2gumbel(u)
    _eps = eps(eltype(u))
    return -log.(-log.(u .+ _eps) .+ _eps)
end

function u2gumbelback(u, g)
    T = eltype(u)
    return one(T) ./ exp.(-g) ./ (u .+ eps(T))
end

u2gumbel(u::Tracker.TrackedArray) = Tracker.track(u2gumbel, u)

Tracker.@grad function u2gumbel(u)
    g = u2gumbel(Tracker.data(u))
    return g, Δ -> (Δ .* u2gumbelback(u, g),)
end

function g2softmax(g, p, τ)
    logit = g .+ log.(p .+ eps(eltype(g)))
    return NNlib.softmax(logit ./ τ; dims=1)
end

function _rand(gs::AbstractGumbelSoftmax, n::Int=1)
    u = randsimilar(gs.p, n)
    g = u2gumbel(u)
    return g2softmax(g, gs.p, gs.τ)
end

rand(gs::AbstractGumbelSoftmax) = _rand(gs)

rand(gs::AbstractGumbelSoftmax{<:AbstractVector}, n::Int) = _rand(gs, n)

mean(gs::AbstractGumbelSoftmax) = gs.p

function GumbelSoftmax2D(p; τ=eltype(p)(τdefault))
    p = transpose(hcat(p, one(eltype(p)) .- p))
    return GumbelSoftmax(p, τ)
end

# """
# The Gumbel-Bernoulli distributions.
#
# NOTE: parameters are in batch.
#
# Ref: https://arxiv.org/abs/1611.01144
# """
# struct GumbelBernoulli{T}
#     p::T
#     τ::FT
# end
#
# GumbelBernoulli(p; τ=FT(τdefault)) = GumbelBernoulli(p, τ)
#
# """
#     rand(gb::GumbelBernoulli)
#
# Sample from Gumbel-Bernoulli distributions.
# """
# function rand(gb::GumbelBernoulli{T}) where {T}
#     # TODO: re-implement this `rand` using the same procedure for `GumbelBernoulliLogit`
#     _eps = eps(eltype(gb.p))
#     _one = one(eltype(gb.p))
#     τ = gb.τ
#
#     u0 = randsimilar(gb.p); g0 = u2gumbel(u0)
#     u1 = randsimilar(gb.p); g1 = u2gumbel(u1)
#
#     logit0 = (g0 .+ log.(_one + _eps .- gb.p)) ./ τ
#     logit1 = (g1 .+ log.(gb.p .+ _eps)) ./ τ
#
#     logit_max = max.(logit0, logit1)
#     logit1_minus_max = logit1 .- logit_max
#
#     logx = logit1_minus_max .- log.(exp.(logit0 - logit_max) .+ exp.(logit1_minus_max))
#     return exp.(logx)
# end
#
# """
#     logpdf(bgb::GumbelBernoulli, x)
#
# Compute ``GumbelBernoulli(x; p)``.
#
# NOTE: `p` and `x` are assumed to be in batch.
#
# Ref: https://arxiv.org/abs/1611.01144
# """
# function logpdf(bgb::GumbelBernoulli, x)
#     _eps = eps(FT)
#     τ = bgb.τ
#     α = bgb.p ./ (1 .- bgb.p .+ _eps)
#     xstabe = x .+ _eps
#     omxstabe = 1 .- x .+ _eps
#     return log(τ) .+ log.(α) + (-τ - 1) * (log.(xstabe) + log.(omxstabe)) - 2 * (log.(α .* xstabe.^(-τ) + omxstabe.^(-τ) .+ _eps))
# end
#
# """
#     logpdfcov(bgb::GumbelBernoulli, x)
#
# Compute ``GumbelBernoulli(x; p)`` by using that of `GumbelBernoulli(logitx; logitp)`
# together with change of variable (CoV) rules.
#
# NOTE: `p` and `x` are assumed to be in batch.
# """
# function logpdfCoV(bgb::GumbelBernoulli, x)
#     _eps = eps(FT)
#     _1m2eps = 1 - 2 * _eps
#     logitp = logit.(bgb.p .* _1m2eps .+ _eps)
#     logitx = logit.(x .* _1m2eps .+ _eps)
#     lp = logpdflogit(GumbelBernoulliLogit(logitp; τ=bgb.τ), logitx)
#     _eps = eps(FT)
#     return lp - log.(x .* (1 .- x) .+ _eps)
# end
#
# mean(gb::GumbelBernoulli) = gb.p
#
# """
# The Gumbel-Bernoulli distributions in logit space.
#
# NOTE: parameters are in batch.
#
# Ref: https://arxiv.org/abs/1611.01144
# """
# struct GumbelBernoulliLogit{T}
#     logitp::T
#     τ::FT
# end
#
# GumbelBernoulliLogit(logitp; τ=FT(τdefault)) = GumbelBernoulliLogit(logitp, τ)
#
# """
#     logitrand(gbl::GumbelBernoulliLogit{T}; τ=gbl.τ) where {T}
#
# Sample logit from Bernoulli distributions by logit.
#
# NOTE: `lp` is assumed to be in batch
#
# Ref: https://arxiv.org/abs/1611.00712
# """
# function logitrand(gbl::GumbelBernoulliLogit{T}; τ=gbl.τ) where {T}
#     FT = eltype(gbl.logitp)
#     _eps = eps(FT)
#     _one = one(FT)
#
#     u = randsimilar(gbl.logitp)
#
#     logit = log.(u .+ _eps) - log.(_one + _eps .- u)
#
#     logitx = (gbl.logitp + logit) ./ τ
#     return logitx
# end
#
# """
#     logpdflogit(gbl::GumbelBernoulliLogit, logitx)
#
# Compute `GumbelBernoulli(logitx; logitp)`.
#
# NOTE: `logitp` and `logitx` are assumed to be in batch.
#
# WARN: this function is not tested.
#
# Ref: https://arxiv.org/abs/1611.00712
# """
# function logpdflogit(gbl::GumbelBernoulliLogit, logitx; τ=gbl.τ)
#     exp_term = gbl.logitp .- logitx .* τ
#     lp = exp_term .+ log(τ) .- FT(2.0) .* softplus.(exp_term)
#     return lp
# end
#
# mean(gbl::GumbelBernoulliLogit) = Knet.sigm.(gbl.logitp)
