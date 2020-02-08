using CuArrays: CuArrays, @cufunc, CuMatOrAdj, CuOrAdj, CUBLAS

### Base.:\

import Base: \

# Tracker

using Tracker: Tracker, TrackedArray, @grad, data

Base.inv(x::CuMatOrAdj{<:AbstractFloat}) = CUBLAS.matinv_batched([x])[2][1]

A::CuMatOrAdj   \ B::TrackedArray = Tracker.track(\, A, B)
A::TrackedArray \ B::CuOrAdj      = Tracker.track(\, A, B)

@grad function (A::Union{CuMatOrAdj,TrackedArray} \ B::Union{CuOrAdj,TrackedArray})
    return data(A) \ data(B), function (Δ)
        AtransposedivΔ = transpose(A) \ Δ
        ∇A = transpose(A \ B * transpose(-AtransposedivΔ))
        return (∇A,  AtransposedivΔ)
    end
end

# Zygote

using Zygote: @adjoint

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
@cufunc logistic(x) = inv(exp(-x) + one(x))
