using CuArrays

Base.inv(x::CuArray{<:Real,2}) = CuArrays.CUBLAS.matinv_batched([x])[2][1]

function Base.:\(_A::CuArray{<:Real}, _B::CuArray{<:Real})
    A, B = copy(_A), copy(_B)
    A, ipiv = CuArrays.CUSOLVER.getrf!(A)
    return CuArrays.CUSOLVER.getrs!('N', A, ipiv, B)
end
