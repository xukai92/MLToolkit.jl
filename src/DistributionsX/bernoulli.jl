### Bernoulli

struct Bernoulli{T<:AbstractArray{<:AbstractFloat}} <: DiscreteBatchDistribution
    p::T
end

Broadcast.broadcastable(b::Bernoulli) = Ref(b)

function logpdf(b::Bernoulli, x)
    ϵ = eps(b.p)
    lp = x .* log.(b.p .+ ϵ) + (1 .- x) .* log.(1 + ϵ .- b.p)
    return lp
end

function rand(rng::AbstractRNG, b::Bernoulli)
    u = randsimilar(rng, b.p)
    x = convert.(eltype(b.p), b.p .> u)
    return x
end

mean(b::Bernoulli) = b.p

function mode(b::Bernoulli)
    T = eltype(b.p)
    return convert.(T, b.p .> T(0.5))
end

function kldiv(
    b1::Bernoulli{<:AbstractArray{T}},
    b2::Bernoulli{<:AbstractArray{T}}
) where {T<:AbstractFloat}
    ϵ = eps(T)
    kl = b1.p .* (log.(b1.p .+ ϵ) .- log.(b2.p .+ ϵ)) .+
         (1 .- b1.p) .* (log.(1 + ϵ .- b1.p) .- log.(1 + ϵ .- b2.p))
    return kl
end

### BernoulliLogit

struct BernoulliLogit{T<:AbstractArray{<:AbstractFloat}} <: DiscreteBatchDistribution
    logitp::T
end

Broadcast.broadcastable(bl::BernoulliLogit) = Ref(bl)

logpdf(bl::BernoulliLogit, x) = x .* bl.logitp .- log.(1 .+ exp.(bl.logitp))

mean(bl::BernoulliLogit) = StatsFuns.logistic.(bl.logitp)

mode(bl::BernoulliLogit) = mode(Bernoulli(mean(bl)))
