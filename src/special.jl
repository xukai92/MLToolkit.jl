using SpecialFunctions: lgamma
import SpecialFunctions: lbeta, beta

# Call back functions for `lbeta` and `beta`
# Those in `SpecialFunctions.jl` are implemented for `(::Number, ::Number)` only,
# for which `KnetArray` cannot broadcast through.
# NOTE: `_lbeta` and `_beta` are implemented for explictly test purpose.
# TODO: implement this broadcasting in Knet.jl
_lbeta(α, β) = lgamma.(α) + lgamma.(β) - lgamma.(α + β)
_beta(α, β) = exp.(_lbeta(α, β))
lbeta(α, β) = _lbeta(α, β)
beta(α, β) = _beta(α, β)

import StatsFuns: logit

function logit(x)
    _eps = eps(FT)
    _one = one(FT)
    return log(x + _eps) - log(_one - x + _eps)
end

function logsumexp2(x::AbstractHigherOrderArray; dims=:)
    u = maximum(x; dims=dims)
    lsediff = log.(sum(exp.(x .- u); dims=dims))
    return u .+ lsediff
end

logsumexp2(x::Tracker.TrackedArray; dims=:) = Tracker.track(logsumexp2, x; dims=dims)

Tracker.@grad function logsumexp2(x::Tracker.TrackedArray; dims=:)
    lse = logsumexp2(Tracker.data(x); dims=dims)
    return lse, Δ -> (Δ .* exp.(x .- lse),)
end
