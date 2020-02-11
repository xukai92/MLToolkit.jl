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

mean_cpu(d::UniformNoise{T}) where {T} = zeros(T, d.size)
mean(d::UniformNoise{T,Val{:cpu}}) where {T} = mean_cpu(d)
var_cpu(d::UniformNoise{T}) where {T} = ones(T, d.size) / 3
var(d::UniformNoise{T,Val{:cpu}}) where {T} = var_cpu(d)

UniformNoise(T, dim::Int) = UniformNoise(T, :cpu, (dim,))
UniformNoise(size::Int...) = UniformNoise(Float64, :cpu, size)

_rand(rng, d::UniformNoise{T}, dims::Int...) where {T} = 2 * rand(rng, T, d.size..., dims...) .- 1

function rand(rng::AbstractRNG, d::UniformNoise{T,Val{:cpu}}, dims::Int...) where {T}
    return _rand(rng, d, dims...)
end

logpdf(d::UniformNoise{T}, x::AbstractArray{T}) where {T} = fill(-log(2one(T)), size(x))

###

CuArrays.cu(x::UniformNoise) = UniformNoise(Float32, :gpu, x.size)

Flux.adapt(::Type{Array}, x::UniformNoise{T,<:Any,S}) where {T,S} = UniformNoise(T, :cpu, x.size)

mean(d::UniformNoise{T,Val{:gpu}}) where {T} = gpu(mean_cpu(d))
var(d::UniformNoise{T,Val{:gpu}}) where {T} = gpu(var_cpu(d))

function rand(rng::AbstractRNG, d::UniformNoise{T,Val{:gpu}}, dims::Int...) where {T}
    return gpu(_rand(rng, d, dims...))
end

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

mean_cpu(d::GaussianNoise{T}) where {T} = zeros(T, d.size)
mean(d::GaussianNoise{T,Val{:cpu}}) where {T} = mean_cpu(d)
var_cpu(d::GaussianNoise{T}) where {T} = ones(T, d.size)
var(d::GaussianNoise{T,Val{:cpu}}) where {T} = var_cpu(d)

GaussianNoise(T, dim::Int) = GaussianNoise(T, :cpu, (dim,))
GaussianNoise(size::Int...) = GaussianNoise(Float64, :cpu, size)

_rand(rng, d::GaussianNoise{T}, dims::Int...) where {T} = randn(rng, T, d.size..., dims...)

function rand(rng::AbstractRNG, d::GaussianNoise{T,Val{:cpu}}, dims::Int...) where {T}
    return _rand(rng, d, dims...)
end

logpdf(d::GaussianNoise{T}, x::AbstractArray{T}) where {T} = -(log(2T(π)) .+ map(abs2, x)) / 2

###

CuArrays.cu(x::GaussianNoise) = GaussianNoise(Float32, :gpu, x.size)

Flux.adapt(::Type{Array}, x::GaussianNoise{T,<:Any,S}) where {T,S} = GaussianNoise(T, :cpu, x.size)

mean(d::GaussianNoise{T,Val{:gpu}}) where {T} = gpu(mean_cpu(d))
var(d::GaussianNoise{T,Val{:gpu}}) where {T} = gpu(var_cpu(d))

function rand(rng::AbstractRNG, d::GaussianNoise{T,Val{:gpu}}, dims::Int...) where {T}
    return gpu(_rand(rng, d, dims...))
end
