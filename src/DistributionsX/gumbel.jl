### GumbelSoftmax

abstract type AbstractGumbelSoftmax{T} <: ContinuousMultivariateDistribution end

const τ0 = 0.2  # default value for τ

"""
    GumbelSoftmax(p, τ)

The Gumbel-Softmax distribution with mean `p` and temperature `τ`.

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

function _rand(rng::AbstractRNG, p, τ, n::Int=1)
    u = randsimilar(rng, p, n)
    g = u2gumbel(u)
    return g2softmax(g, p, τ)
end

rand(
    rng::AbstractRNG,
    gs::AbstractGumbelSoftmax;
    τ=gs.τ
) = _rand(rng, gs.p, τ)

rand(
    rng::AbstractRNG,
    gs::AbstractGumbelSoftmax{<:AbstractVector},
    n::Int;
    τ=gs.τ
) = _rand(rng, gs.p, τ, n)

mean(gs::AbstractGumbelSoftmax) = gs.p

"""
    GumbelSoftmax2D(p1, τ)

2-dimensional GumbelSoftmax where `p1` is the probability of the first dimension is 1.
"""
function GumbelSoftmax2D(p1::AbstractVector; τ=eltype(p1)(τ0))
    p1transpose = transpose(p1)
    p = [p1transpose; 1 .- p1transpose]
    return GumbelSoftmax(p, τ)
end
function GumbelSoftmax2D(p1::T; τ=T(τ0)) where {T<:AbstractFloat}
    return GumbelSoftmax([p1, 1 - p1], τ)
end

### GumbelBernoulli

abstract type AbstractGumbelBernoulli{T} <: ContinuousMultivariateDistribution end

"""
The Gumbel-Bernoulli distribution with mean `p` and temperature `τ`.

Ref: https://arxiv.org/abs/1611.01144
"""
struct GumbelBernoulli{T<:AbstractArray{<:AbstractFloat}} <: AbstractGumbelBernoulli{T}
    p::T
    τ
    function GumbelBernoulli(p::T, τ::F) where {F<:AbstractFloat, T<:AbstractArray{F}}
        return new{T}(p, τ)
    end
end

GumbelBernoulli(p; τ=eltype(p)(τ0)) = GumbelBernoulli(p, τ)

"""
    logrand(rng, gb)

Generate sample in log space from a Gumbel-Bernoulli distribution.

Ref: https://arxiv.org/abs/1611.00712
"""
function logrand(rng::AbstractRNG, gb::GumbelBernoulli; τ=gb.τ)
    # TODO: re-implement this `rand` using the same procedure for `GumbelBernoulliLogit`
    u0 = randsimilar(rng, gb.p); g0 = u2gumbel(u0)
    u1 = randsimilar(rng, gb.p); g1 = u2gumbel(u1)

    ϵ = eps(gb.p)

    logit0 = (g0 + log.(1 + ϵ .- gb.p)) / τ
    logit1 = (g1 + log.(gb.p .+ ϵ)) / τ

    logit_max = max.(logit0, logit1)
    logit0_minus_max = logit0 - logit_max
    logit1_minus_max = logit1 - logit_max

    logx = logit1_minus_max - log.(exp.(logit0_minus_max) + exp.(logit1_minus_max))
    return logx
end
logrand(gb::GumbelBernoulli; τ=gb.τ) = logrand(GLOBAL_RNG, gb; τ=τ)

rand(rng::AbstractRNG, gb::GumbelBernoulli; τ=gb.τ) = exp.(logrand(rng, gb; τ=τ))

function _logpdf(gb::GumbelBernoulli, x; τ=gb.τ)
    ϵ = eps(gb.p)
    α = gb.p ./ (1 + ϵ .- gb.p)
    xstabe = x .+ ϵ
    omxstabe = 1 .- x .+ ϵ
    return log(τ) .+ log.(α) + (-τ - 1) * (log.(xstabe) .+ log.(omxstabe)) - 2 * (log.(α .* xstabe.^(-τ) .+ omxstabe.^(-τ) .+ ϵ))
end

logpdf(gb::GumbelBernoulli, x; τ=gb.τ) = _logpdf(gb, x; τ=τ)
logpdf(gb::GumbelBernoulli, x::AbstractMatrix{<:AbstractFloat}; τ=gb.τ) = _logpdf(gb, x; τ=τ)

"""
    logpdfCoV(bgb, x)

Compute the log density of `GumbelBernoulli(x; p)` by using
that of `GumbelBernoulli(logitx; logitp)` by the change of variable (CoV) rule.
"""
function logpdfCoV(gb::GumbelBernoulli, x; τ=gb.τ)
    ϵ = eps(gb.p)
    om2ϵ = 1 - 2 * ϵ
    logitp = StatsFuns.logit.(gb.p * om2ϵ .+ ϵ)
    logitx = StatsFuns.logit.(x * om2ϵ .+ ϵ)
    lp = logpdflogit(GumbelBernoulliLogit(logitp; τ=τ), logitx)
    return lp - log.(x .* (1 .- x) .+ ϵ)
end

mean(gb::GumbelBernoulli) = gb.p

"""
    GumbelBernoulliLogit(p, τ)

The Gumbel-Softmax distribution with mean `logitp` in logit space and temperature `τ`.

Ref: https://arxiv.org/abs/1611.01144
"""
struct GumbelBernoulliLogit{T} <: AbstractGumbelBernoulli{T}
    logitp::T
    τ
    function GumbelBernoulliLogit(logitp::T, τ::F) where {F<:AbstractFloat, T<:AbstractArray{F}}
        return new{T}(logitp, τ)
    end
end

GumbelBernoulliLogit(p; τ=eltype(p)(τ0)) = GumbelBernoulliLogit(p, τ)

"""
    logitrand(rng, gbl)

Generate sample in logit space from a Gumbel-Bernoulli distribution
with parameters in logit space.

Ref: https://arxiv.org/abs/1611.00712
"""
function logitrand(rng::AbstractRNG, gbl::GumbelBernoulliLogit; τ=gbl.τ)
    ϵ = eps(gbl.logitp)
    u = randsimilar(rng, gbl.logitp)
    logit = log.(u .+ ϵ) - log.(1 + ϵ .- u)
    logα = gbl.logitp
    logitx = (logα + logit) ./ τ
    return logitx
end
logitrand(gbl::GumbelBernoulliLogit; τ=gbl.τ) = logitrand(GLOBAL_RNG, gbl; τ=τ)

rand(
    rng::AbstractRNG,
    gbl::GumbelBernoulliLogit;
    τ=gbl.τ
) = StatsFuns.logistic.(logitrand(rng, gbl; τ=τ))

"""
    logpdflogit(gbl, logitx)

Compute the log density of `GumbelBernoulli(logitx; logitp)`.

Ref: https://arxiv.org/abs/1611.00712
"""
function logpdflogit(gbl::GumbelBernoulliLogit, logitx; τ=gbl.τ)
    exp_term = gbl.logitp .- τ * logitx
    lp = exp_term - 2 * StatsFuns.softplus.(exp_term) .+ log(τ)
    return lp
end

mean(gbl::GumbelBernoulliLogit) = StatsFuns.logistic.(gbl.logitp)
