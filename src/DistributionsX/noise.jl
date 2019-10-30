### Uniform(-1, 1)

struct UniformNoise{T<:AbstractFloat,S<:Tuple{Vararg{Int}}} <: ContinuousBatchDistribution
    size::S
    UniformNoise(T::Type{<:AbstractFloat}, size::S) where {S<:Tuple{Vararg{Int}}} = new{T,S}(size)
end

mean(d::UniformNoise{T}) where {T} = zeros(T, d.size)
var(d::UniformNoise{T}) where {T} = ones(T, d.size) / 3

UniformNoise(T, dim::Int) = UniformNoise(T, (dim,))
UniformNoise(size) = UniformNoise(Float64, size)

function rand(rng::AbstractRNG, d::UniformNoise{T}, n::Int) where {T}
    return 2 * rand(rng, T, d.size..., n) .- 1
end

_logpdf_vec(d::UniformNoise{T}) where {T} = -prod(d.size) * log(2one(T))
logpdf(d::UniformNoise{T}, ::AbstractVector{T}) where {T} = _logpdf_vec(d)
logpdf(d::UniformNoise{T}, x::AbstractArray{T}) where {T} = fill(_logpdf_vec(d), last(size(x)))

### Standard Normal

struct GaussianNoise{T<:AbstractFloat,S<:Tuple{Vararg{Int}}} <: ContinuousBatchDistribution
    size::S
    GaussianNoise(T::Type{<:AbstractFloat}, size::S) where {S<:Tuple{Vararg{Int}}} = new{T,S}(size)
end

mean(d::GaussianNoise{T}) where {T} = zeros(T, d.size)
var(d::GaussianNoise{T}) where {T} = ones(T, d.size)

GaussianNoise(T, dim::Int) = GaussianNoise(T, (dim,))
GaussianNoise(size) = GaussianNoise(Float64, size)

rand(rng::AbstractRNG, d::GaussianNoise{T}, n::Int) where {T} = randn(rng, T, d.size..., n)

_constant(d::GaussianNoise{T}) where {T} = prod(d.size) * log(2T(π))
logpdf(d::GaussianNoise{T}, x::AbstractVector{T})  where {T}   = -(_constant(d)  + sum(abs2, x)) / 2
logpdf(d::GaussianNoise{T}, x::AbstractArray{T,N}) where {T,N} = -(_constant(d) .+ vec(sum(abs2, x; dims=1:N-1))) / 2
