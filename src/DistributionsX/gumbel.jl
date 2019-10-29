abstract type AbstractGumbelSoftmax{T} <: ContinuousMultivariateDistribution end

const τ0 = 0.2  # default value for τ

"""
    GumbelSoftmax(p, τ)

The Gumbel-Softmax distributions with mean `p` and temperature `τ`.

Ref: https://arxiv.org/abs/1611.01144
"""
struct GumbelSoftmax{T<:AbstractVecOrMat{<:AbstractFloat}} <: AbstractGumbelSoftmax{T}
    p::T
    τ
    function GumbelSoftmax(p::T, τ::F) where {F<:AbstractFloat, T<:AbstractVecOrMat{F}}
        return new{T}(p, τ)
    end
end

GumbelSoftmax(p; τ=eltype(p)(τ0)) = GumbelSoftmax(p, τ)

function u2gumbel(u)
    ϵ = eps(u)
    return -log.(-log.(u .+ ϵ) .+ ϵ)
end

function u2gumbelback(u, g)
    return 1 ./ exp.(-g) ./ (u .+ eps(u))
end

u2gumbel(u::Tracker.TrackedArray) = Tracker.track(u2gumbel, u)

Tracker.@grad function u2gumbel(u)
    g = u2gumbel(Tracker.data(u))
    return g, Δ -> (Δ .* u2gumbelback(u, g),)
end

function g2softmax(g, p, τ)
    logit = g .+ log.(p .+ eps(g))
    return NNlib.softmax(logit ./ τ; dims=1)
end

function _rand(rng::AbstractRNG, gs::AbstractGumbelSoftmax, n::Int=1)
    u = randsimilar(rng, gs.p, n)
    g = u2gumbel(u)
    return g2softmax(g, gs.p, gs.τ)
end

rand(rng::AbstractRNG, gs::AbstractGumbelSoftmax) = _rand(rng, gs)

rand(rng::AbstractRNG, gs::AbstractGumbelSoftmax{<:AbstractVector}, n::Int) = _rand(rng, gs, n)

mean(gs::AbstractGumbelSoftmax) = gs.p

"""
    GumbelSoftmax2D(p1, τ)

2-dimensional GumbelSoftmax where `p1` is the probability of the first dimension is 1.
"""
function GumbelSoftmax2D(p1::AbstractVector; τ=eltype(p1)(τ0))
    p = transpose(hcat(p1, one(eltype(p1)) .- p1))
    return GumbelSoftmax(p, τ)
end
function GumbelSoftmax2D(p1::T; τ=T(τ0)) where {T<:AbstractFloat}
    return GumbelSoftmax([p1, 1 - p1], τ)
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
# GumbelBernoulli(p; τ=FT(τ0)) = GumbelBernoulli(p, τ)
#
# """
#     rand(gb::GumbelBernoulli)
#
# Sample from Gumbel-Bernoulli distributions.
# """
# function rand(gb::GumbelBernoulli{T}) where {T}
#     # TODO: re-implement this `rand` using the same procedure for `GumbelBernoulliLogit`
#     ϵ = eps(eltype(gb.p))
#     _one = one(eltype(gb.p))
#     τ = gb.τ
#
#     u0 = randsimilar(gb.p); g0 = u2gumbel(u0)
#     u1 = randsimilar(gb.p); g1 = u2gumbel(u1)
#
#     logit0 = (g0 .+ log.(_one + ϵ .- gb.p)) ./ τ
#     logit1 = (g1 .+ log.(gb.p .+ ϵ)) ./ τ
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
#     ϵ = eps(FT)
#     τ = bgb.τ
#     α = bgb.p ./ (1 .- bgb.p .+ ϵ)
#     xstabe = x .+ ϵ
#     omxstabe = 1 .- x .+ ϵ
#     return log(τ) .+ log.(α) + (-τ - 1) * (log.(xstabe) + log.(omxstabe)) - 2 * (log.(α .* xstabe.^(-τ) + omxstabe.^(-τ) .+ ϵ))
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
#     ϵ = eps(FT)
#     _1m2eps = 1 - 2 * ϵ
#     logitp = logit.(bgb.p .* _1m2eps .+ ϵ)
#     logitx = logit.(x .* _1m2eps .+ ϵ)
#     lp = logpdflogit(GumbelBernoulliLogit(logitp; τ=bgb.τ), logitx)
#     ϵ = eps(FT)
#     return lp - log.(x .* (1 .- x) .+ ϵ)
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
# GumbelBernoulliLogit(logitp; τ=FT(τ0)) = GumbelBernoulliLogit(logitp, τ)
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
#     ϵ = eps(FT)
#     _one = one(FT)
#
#     u = randsimilar(gbl.logitp)
#
#     logit = log.(u .+ ϵ) - log.(_one + ϵ .- u)
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
