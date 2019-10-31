### Uniform(-1, 1)

struct UniformNoise{
    T<:AbstractFloat,
    D<:Union{Val{:cpu},Val{:gpu}},
    S<:Tuple{Vararg{Int}}
} <: ContinuousBatchDistribution
    size::S
    function UniformNoise(T::Type{<:AbstractFloat}, device::Symbol, size::S) where {S<:Tuple{Vararg{Int}}}
        return new{T,Val{device},S}(size)
    end
end

Base.size(d::UniformNoise) = d.size

_mean(d::UniformNoise{T}) where {T} = zeros(T, d.size)
mean(d::UniformNoise{T,Val{:cpu}}) where {T} = _mean(d)
_var(d::UniformNoise{T}) where {T} = ones(T, d.size) / 3
var(d::UniformNoise{T,Val{:cpu}}) where {T} = _var(d)

UniformNoise(T, dim::Int) = UniformNoise(T, :cpu, (dim,))
UniformNoise(size::Int...) = UniformNoise(Float64, :cpu, size)

_rand(rng, d::UniformNoise{T}, dims::Int...) where {T} = 2 * rand(rng, T, d.size..., dims...) .- 1

function rand(rng::AbstractRNG, d::UniformNoise{T,Val{:cpu}}, dims::Int...) where {T}
    return _rand(rng, d, dims...)
end

logpdf(d::UniformNoise{T}, x::AbstractArray{T}) where {T} = fill(-log(2one(T)), size(x))

### Standard Normal

struct GaussianNoise{
    T<:AbstractFloat,
    D<:Union{Val{:cpu},Val{:gpu}},
    S<:Tuple{Vararg{Int}}
} <: ContinuousBatchDistribution
    size::S
    function GaussianNoise(T::Type{<:AbstractFloat}, device::Symbol, size::S) where {S<:Tuple{Vararg{Int}}}
        return new{T,Val{device},S}(size)
    end
end

Base.size(d::GaussianNoise) = d.size

_mean(d::GaussianNoise{T}) where {T} = zeros(T, d.size)
mean(d::GaussianNoise{T,Val{:cpu}}) where {T} = _mean(d)
_var(d::GaussianNoise{T}) where {T} = ones(T, d.size)
var(d::GaussianNoise{T,Val{:cpu}}) where {T} = _var(d)

GaussianNoise(T, dim::Int) = GaussianNoise(T, :cpu, (dim,))
GaussianNoise(size::Int...) = GaussianNoise(Float64, :cpu, size)

_rand(rng, d::GaussianNoise{T}, dims::Int...) where {T} = randn(rng, T, d.size..., dims...)

function rand(rng::AbstractRNG, d::GaussianNoise{T,Val{:cpu}}, dims::Int...) where {T}
    return _rand(rng, d, dims...)
end

logpdf(d::GaussianNoise{T}, x::AbstractArray{T}) where {T} = -(log(2T(π)) .+ map(abs2, x)) / 2
