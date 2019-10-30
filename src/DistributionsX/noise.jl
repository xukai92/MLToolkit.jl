### Uniform(-1, 1)

struct UniformNoise{
    T<:AbstractFloat,
    S<:Tuple{Vararg{Int}},
    D<:Union{Val{:cpu},Val{:gpu}}
} <: ContinuousBatchDistribution
    size::S
    function UniformNoise(T::Type{<:AbstractFloat}, size::S, device::Symbol) where {S<:Tuple{Vararg{Int}}}
        return new{T,S,Val{device}}(size)
    end
end

Base.size(d::UniformNoise) = d.size

mean(d::UniformNoise{T}) where {T} = zeros(T, d.size)
var(d::UniformNoise{T}) where {T} = ones(T, d.size) / 3

UniformNoise(T, dim::Int) = UniformNoise(T, (dim,), :cpu)
UniformNoise(size::Int...) = UniformNoise(Float64, size, :cpu)

_rand(rng, d::UniformNoise{T}, dims::Int...) where {T} = 2 * rand(rng, T, d.size..., dims...) .- 1

function rand(rng::AbstractRNG, d::UniformNoise{T,S,Val{:cpu}}, dims::Int...) where {T,S}
    return _rand(rng, d, dims...)
end

logpdf(d::UniformNoise{T}, x::AbstractArray{T}) where {T} = fill(-log(2one(T)), size(x))

### Standard Normal

struct GaussianNoise{
    T<:AbstractFloat,
    S<:Tuple{Vararg{Int}},
    D<:Union{Val{:cpu},Val{:gpu}}
} <: ContinuousBatchDistribution
    size::S
    function GaussianNoise(T::Type{<:AbstractFloat}, size::S, device::Symbol) where {S<:Tuple{Vararg{Int}}}
        return new{T,S,Val{device}}(size)
    end
end

Base.size(d::GaussianNoise) = d.size

mean(d::GaussianNoise{T}) where {T} = zeros(T, d.size)
var(d::GaussianNoise{T}) where {T} = ones(T, d.size)

GaussianNoise(T, dim::Int) = GaussianNoise(T, (dim,), :cpu)
GaussianNoise(size::Int...) = GaussianNoise(Float64, size, :cpu)

_rand(rng, d::GaussianNoise{T}, dims::Int...) where {T} = randn(rng, T, d.size..., dims...)

function rand(rng::AbstractRNG, d::GaussianNoise{T,S,Val{:cpu}}, dims::Int...) where {T,S}
    return _rand(rng, d, dims...)
end

logpdf(d::GaussianNoise{T}, x::AbstractArray{T}) where {T} = -(log(2T(π)) .+ map(abs2, x)) / 2
