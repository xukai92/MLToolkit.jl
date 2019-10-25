using CuArrays

CuMatOrAdj{T} = Union{CuMatrix, LinearAlgebra.Adjoint{T, CuMatrix{T}}, LinearAlgebra.Transpose{T, CuMatrix{T}}}
CuOrAdj{T} = Union{CuVecOrMat, LinearAlgebra.Adjoint{T, CuVecOrMat{T}}, LinearAlgebra.Transpose{T, CuVecOrMat{T}}}

Base.inv(x::CuMatOrAdj{<:AbstractFloat}) = CuArrays.CUBLAS.matinv_batched([x])[2][1]

Tracker.@grad function (A::CuMatOrAdj{T} \ B::CuOrAdj{T}) where {T<:AbstractFloat}
    return Tracker.data(A) \ Tracker.data(B), function (Δ)
        ∇A = -(A' \ Δ) * B' / A'
        return (∇A,  (A' \ Δ))
    end
end
