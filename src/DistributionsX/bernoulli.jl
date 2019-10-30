### BatchBernoulli

struct BatchBernoulli{T<:AbstractArray{<:AbstractFloat}} <: DiscreteBatchDistribution
    p::T
end

Broadcast.broadcastable(b::BatchBernoulli) = Ref(b)

Distributions.Bernoulli(p::AbstractArray) = BatchBernoulli(p)

function logpdf(b::BatchBernoulli, x)
    ϵ = eps(b.p)
    lp = x .* log.(b.p .+ ϵ) + (1 .- x) .* log.(1 + ϵ .- b.p)
    return lp
end

function rand(rng::AbstractRNG, b::BatchBernoulli, dims::Int...)
    u = randsimilar(rng, b.p, dims...)
    x = convert.(eltype(b.p), b.p .> u)
    return x
end

mean(b::BatchBernoulli) = b.p

function var(b::BatchBernoulli)
    p = mean(b)
    return p .* (1 .- p)
end

function mode(b::BatchBernoulli)
    T = eltype(b.p)
    return convert.(T, b.p .> T(0.5))
end

function kldiv(
    b1::BatchBernoulli{<:AbstractArray{T}},
    b2::BatchBernoulli{<:AbstractArray{T}}
) where {T<:AbstractFloat}
    ϵ = eps(T)
    kl = b1.p .* (log.(b1.p .+ ϵ) .- log.(b2.p .+ ϵ)) .+
         (1 .- b1.p) .* (log.(1 + ϵ .- b1.p) .- log.(1 + ϵ .- b2.p))
    return kl
end

### BatchBernoulliLogit

struct BatchBernoulliLogit{T<:AbstractArray{<:AbstractFloat}} <: DiscreteBatchDistribution
    logitp::T
end

Broadcast.broadcastable(bl::BatchBernoulliLogit) = Ref(bl)

BernoulliLogit(logitp) = BatchBernoulliLogit(logitp)

logpdf(bl::BatchBernoulliLogit, x) = x .* bl.logitp .- log.(1 .+ exp.(bl.logitp))

mean(bl::BatchBernoulliLogit) = StatsFuns.logistic.(bl.logitp)

function var(b::BatchBernoulliLogit)
    p = mean(b)
    return p .* (1 .- p)
end

mode(bl::BatchBernoulliLogit) = mode(Bernoulli(mean(bl)))
