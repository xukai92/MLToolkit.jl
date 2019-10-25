using CuArrays
using CuArrays: CuMatOrAdj, CuOrAdj

Base.inv(x::CuMatOrAdj{<:AbstractFloat}) = CuArrays.CUBLAS.matinv_batched([x])[2][1]

import Base.\
A::Tracker.TrackedArray \ B::Tracker.TrackedArray = Tracker.track(\, A, B)
A::CuMatOrAdj           \ B::Tracker.TrackedArray = Tracker.track(\, A, B)
A::Tracker.TrackedArray \ B::CuOrAdj              = Tracker.track(\, A, B)
Tracker.@grad function (A::Union{CuMatOrAdj,Tracker.TrackedArray} \ B::Union{CuOrAdj,Tracker.TrackedArray})
    return Tracker.data(A) \ Tracker.data(B), function (Δ)
        AtransposedivΔ = transpose(A) \ Δ
        ∇A = transpose(A \ B * transpose(-AtransposedivΔ))
        return (∇A,  AtransposedivΔ)
    end
end
