using Bijectors: Bijectors, Bijector, Inversed
using StatsFuns: logit, logistic

struct BroadcastedLogit{T, N} <: Bijector{N}
    a::T
    b::T
end

function BroadcastedLogit(a::T, b::T) where {T}
    return BroadcastedLogit{T, ndims(a)}(a, b)
end

(b::BroadcastedLogit)(x) = logit.((x .- b.a) ./ (b.b - b.a))
(ib::Inversed{<:BroadcastedLogit})(y) = (ib.orig.b - ib.orig.a) .* logistic.(y) .+ ib.orig.a

broadcasted_logabsdetjac(b::BroadcastedLogit, x) = @. - log((x - b.a) * (b.b - x) / (b.b - b.a))

function Bijectors.logabsdetjac(b::BroadcastedLogit{T1}, x::T2) where {T1<:AbstractVector, T2<:AbstractMatrix}
    return dropdims(sum(broadcasted_logabsdetjac(b, x); dims=1); dims=1)
end

###

import Flux
Flux.@functor BroadcastedLogit
