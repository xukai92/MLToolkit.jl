# TODO: implement CUDA kernels for below

log1pexp(x) = log1p(exp(x))
logexpm1(x) = log(-expm1(x))

const softplus = log1pexp
const invsoftplus = logexpm1

function leaky_relu(x; alpha=0.2)
    neg = min(0, x) * alpha
    pos = max(0, x)
    return neg + pos
end

# TODO: merge this with `special.jl`

# import StatsFuns: log1pexp, logexpm1

# # This function is adapted from 
# # https://github.com/denizyuret/Knet.jl/blob/687dba214c3f3326272c7c2191e14e8fdbc35c6a/src/unary.jl#L7-L28
# function unary_op(f)
#     J = Symbol(f)
#     M = which(@__MODULE__, J)
#     @eval begin
#         function Base.Broadcast.broadcasted(::typeof($J), x::Knet.KnetArray{T}) where {T<:AbstractFloat}
#             y = similar(x)
#             @inbounds for i=1:length(y)
#                 xi = x[i]
#                 y[i] = $J(xi)
#             end
#             return y
#         end
#         # TODO: see if I can implement gradient as well; ref:
#         # https://github.com/denizyuret/Knet.jl/blob/687dba214c3f3326272c7c2191e14e8fdbc35c6a/src/unary.jl#L62-L70
#         # Bcasted methods
#         ($M).$J(x::Knet.Bcasted{<:Knet.KnetArray{T}}) where {T<:AbstractFloat} = Base.Broadcast.broadcasted($J, x.value) |> Knet.Bcasted
#         Base.Broadcast.broadcasted(::typeof($J), x::Knet.Bcasted{<:Knet.KnetArray{T}}) where {T<:AbstractFloat} = Base.Broadcast.broadcasted($J, x.value) |> Knet.Bcasted
#     end
#     @eval begin # so we do not trigger some default Base implementation
#         ($M).$J(x::Knet.Bcasted) = throw(MethodError($J,(x,)))
#         Base.Broadcast.broadcasted(::typeof($J), x::Knet.Bcasted) = throw(MethodError($J,(x,)))
#     end
# end

# for f in [:log1pexp, :logexpm1]
#     unary_op(f)
# end

# Knet.AutoGrad.@primitive log1pexp(x::Array),dy,y  dy.*exp.(x .- y)
# Knet.AutoGrad.@primitive log1pexp(x::Knet.KnetArray),dy,y  dy.*exp.(x .- y)

# const softplus = log1pexp
# const invsoftplus = logexpm1