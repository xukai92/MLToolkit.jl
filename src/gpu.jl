using CuArrays
using CuArrays: @cufunc, CuMatOrAdj, CuOrAdj
import Tracker
using Tracker: TrackedArray, @grad, data

### Base

Base.inv(x::CuMatOrAdj{<:AbstractFloat}) = CuArrays.CUBLAS.matinv_batched([x])[2][1]

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

### StatsFuns

import StatsFuns: logit, logistic, log1pexp, logexpm1

@cufunc logit(x) = log(x / (one(x) - x))
@cufunc logistic(x) = inv(exp(-x) + one(x))
@cufunc log1pexp(x) = x < 9  ? log1p(exp(x)) : x < 16  ? x + exp(-x) : x
@cufunc logexpm1(x) = x <= 9 ? log(expm1(x)) : x <= 16 ? x - exp(-x) : x

