using CuArrays

Base.inv(x::CuArray{T,2}) where {T} = CuArrays.CUBLAS.matinv_batched([x])[2][1]