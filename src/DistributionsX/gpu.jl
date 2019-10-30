using CuArrays: Cuarrays, CuArray, cu

# NOTE: The two functions below achive the following behaviour
# - Pass `rng` if it's of type `CuArrays.CURAND.RNG`;
# - Ignore `rng` and use the global of `CuArrays.CURAND` otherwise.
# The motivation is to avoid scalar operations on GPUs, which is the case when
# a CPU's RNG is used for inplace random number generation on GPUs.
rsimilar(rng::CuArrays.CURAND.RNG, f!, x::CuArray, n) = _rsimilar(rng, f!, x, n)
rsimilar(::AbstractRNG, f!, x::CuArray, n) = _rsimilar(CuArrays.CURAND.generator(), f!, x, n)

###

Cuarrays.cu(x::UniformNoise{T}) where {T} = UniformNoise(T, :gpu, x.size)

Flux.adapt(::Type{Array}, x::UniformNoise{T,S}) where {T,S} = UniformNoise{T,S,Val{:cpu}}(x.size)

function rand(rng::AbstractRNG, d::UniformNoise{T,Val{:gpu}}, dims::Int...) where {T}
    return _rand(rng, d, dims) |> cu
end

Cuarrays.cu(x::GaussianNoise{T}) where {T} = GaussianNoise(T, :gpu, x.size)

Flux.adapt(::Type{Array}, x::GaussianNoise{T,S}) where {T,S} = GaussianNoise{T,S,Val{:cpu}}(x.size)

function rand(rng::AbstractRNG, d::GaussianNoise{T,Val{:gpu}}, dims::Int...) where {T}
    return _rand(rng, d, dims) |> cu
end
