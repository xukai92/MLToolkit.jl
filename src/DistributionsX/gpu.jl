using Flux.CuArrays: CuArray, cu, CURAND

# NOTE: The two functions below achive the following behaviour
# - Pass `rng` if it's of type `CuArrays.CURAND.RNG`;
# - Ignore `rng` and use the global of `CuArrays.CURAND` otherwise.
# The motivation is to avoid scalar operations on GPUs, which is the case when
# a CPU's RNG is used for inplace random number generation on GPUs.

rsimilar(rng::CURAND.RNG, f!, x::CuArray, dims::Int...) = _rsimilar(rng, f!, x, dims...)

rsimilar(::AbstractRNG, f!, x::CuArray, dims::Int...) = _rsimilar(CURAND.generator(), f!, x, dims...)

# Rse bleow if we want rsimilar to be reproducible
# TODO: add a global switch
#  rsimilar(rng::AbstractRNG, f, x::CuArray, dims::Int...) = _rsimilar(rng, f, x, dims...) |> cu

### Noise

_rand_nograd_gpu(rng, d, dims...) = _rand(rng, d, dims...) |> gpu
Flux.Zygote.@nograd _rand_nograd_gpu

Flux.CuArrays.cu(x::UniformNoise) = UniformNoise(Float32, :gpu, x.size)

Flux.adapt(::Type{Array}, x::UniformNoise{T,<:Any,S}) where {T,S} = UniformNoise(T, :cpu, x.size)

mean(d::UniformNoise{T,Val{:gpu}}) where {T} = _mean(d) |> cu
var(d::UniformNoise{T,Val{:gpu}}) where {T} = _var(d) |> cu

function rand(rng::AbstractRNG, d::UniformNoise{T,Val{:gpu}}, dims::Int...) where {T}
    return _rand_nograd_gpu(rng, d, dims...)
end

Flux.CuArrays.cu(x::GaussianNoise) = GaussianNoise(Float32, :gpu, x.size)

Flux.adapt(::Type{Array}, x::GaussianNoise{T,<:Any,S}) where {T,S} = GaussianNoise(T, :cpu, x.size)

mean(d::GaussianNoise{T,Val{:gpu}}) where {T} = _mean(d) |> cu
var(d::GaussianNoise{T,Val{:gpu}}) where {T} = _var(d) |> cu

function rand(rng::AbstractRNG, d::GaussianNoise{T,Val{:gpu}}, dims::Int...) where {T}
    return _rand_nograd_gpu(rng, d, dims...)
end

### Normal
