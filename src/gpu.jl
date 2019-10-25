using CuArrays

CuMatOrAdj{T} = Union{CuMatrix, LinearAlgebra.Adjoint{T, <:CuMatrix{T}}, LinearAlgebra.Transpose{T, <:CuMatrix{T}}}
CuOrAdj{T} = Union{CuVecOrMat, LinearAlgebra.Adjoint{T, <:CuVecOrMat{T}}, LinearAlgebra.Transpose{T, <:CuVecOrMat{T}}}

Base.inv(x::CuMatOrAdj{<:AbstractFloat}) = CuArrays.CUBLAS.matinv_batched([x])[2][1]

# TODO: remove this when my PR for CuArrays is merged
function Base.:\(_A::CuMatOrAdj, _B::CuOrAdj)
    A, B = copy(_A), copy(_B)
    A, ipiv = CuArrays.CUSOLVER.getrf!(A)
    return CuArrays.CUSOLVER.getrs!('N', A, ipiv, B)
end

A::Trakcer.TrackedArray \ B::Trakcer.TrackedArray = Tracker.track(\, A, B)
A::CuMatOrAdj           \ B::Trakcer.TrackedArray = Tracker.track(\, A, B)
A::Trakcer.TrackedArray \ B::CuOrAdj              = Tracker.track(\, A, B)
Tracker.@grad function (A::Union{CuMatOrAdj,Trakcer.TrackedArray} \ B::Union{CuMatOrAdj,Trakcer.TrackedArray})
    return Tracker.data(A) \ Tracker.data(B), function (Δ)
        ∇A = -(A' \ Δ) * B' / A'
        return (∇A,  (A' \ Δ))
    end
end
