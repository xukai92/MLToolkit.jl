### Uniform(-1, 1)

struct UniformNoise{T<:AbstractFloat,S<:Tuple{Vararg{Int}}} <: ContinuousBatchDistribution
    size::S
    UniformNoise(T::Type{<:AbstractFloat}, size::S) where {S<:Tuple{Vararg{Int}}} = new{T,S}(size)
end

Base.size(d::UniformNoise) = d.size

mean(d::UniformNoise{T}) where {T} = zeros(T, d.size)
var(d::UniformNoise{T}) where {T} = ones(T, d.size) / 3

UniformNoise(T, dim::Int) = UniformNoise(T, (dim,))
UniformNoise(size::Int...) = UniformNoise(Float64, size)

function rand(rng::AbstractRNG, d::UniformNoise{T}, dims::Int...) where {T}
    return 2 * rand(rng, T, d.size..., dims...) .- 1
end

logpdf(d::UniformNoise{T}, x::AbstractArray{T}) where {T} = fill(-log(2one(T)), size(x))

### Standard Normal

struct GaussianNoise{T<:AbstractFloat,S<:Tuple{Vararg{Int}}} <: ContinuousBatchDistribution
    size::S
    GaussianNoise(T::Type{<:AbstractFloat}, size::S) where {S<:Tuple{Vararg{Int}}} = new{T,S}(size)
end

Base.size(d::GaussianNoise) = d.size

mean(d::GaussianNoise{T}) where {T} = zeros(T, d.size)
var(d::GaussianNoise{T}) where {T} = ones(T, d.size)

GaussianNoise(T, dim::Int) = GaussianNoise(T, (dim,))
GaussianNoise(size::Int...) = GaussianNoise(Float64, size)

rand(rng::AbstractRNG, d::GaussianNoise{T}, dims::Int...) where {T} = randn(rng, T, d.size..., dims...)

logpdf(d::GaussianNoise{T}, x::AbstractArray{T}) where {T} = -(log(2T(π)) .+ map(abs2, x)) / 2
