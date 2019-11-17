using CuArrays
using CuArrays: @cufunc, CuMatOrAdj, CuOrAdj, CUBLAS
import Tracker
using Tracker: TrackedArray, @grad, data
using Flux.Zygote: @adjoint

### Base

Base.inv(x::CuMatOrAdj{<:AbstractFloat}) = CUBLAS.matinv_batched([x])[2][1]

import Base: \

A::CuMatOrAdj   \ B::TrackedArray = Tracker.track(\, A, B)
A::TrackedArray \ B::CuOrAdj      = Tracker.track(\, A, B)

@grad function (A::Union{CuMatOrAdj,TrackedArray} \ B::Union{CuOrAdj,TrackedArray})
    return data(A) \ data(B), function (Δ)
        AtransposedivΔ = transpose(A) \ Δ
        ∇A = transpose(A \ B * transpose(-AtransposedivΔ))
        return (∇A,  AtransposedivΔ)
    end
end

@adjoint Base.:\(A::CuMatOrAdj, B::CuOrAdj) = A \ B, function (Δ)
    AtransposedivΔ = transpose(A) \ Δ
    ∇A = transpose(A \ B * transpose(-AtransposedivΔ))
    return (∇A,  AtransposedivΔ)
end

### Flux

using NNlib: logσ
import Flux: logitbinarycrossentropy

@cufunc logitbinarycrossentropy(logitŷ, y) = (y - 1) * logitŷ - logσ(logitŷ)

### StatsFuns

import StatsFuns: logit, logistic, log1pexp, logexpm1

@cufunc logit(x) = log(x / (one(x) - x))
