module DistributionsX

using Random: AbstractRNG, GLOBAL_RNG, rand!, randn!
import Requires, StatsFuns, NNlib

### Utility

Base.eps(x::AbstractArray) = eps(eltype(x))
Base.one(x::AbstractArray) = one(eltype(x))

"""
    rsimilar(rng, f!, x, n)

Generate random numbers in a container similar to `x`.
The returned variable will have the same shape of `x` if `n == 1` and
will extend an extra dimension with size `n` if `n != 1`.
"""
function _rsimilar(rng::AbstractRNG, f!::Function, x::AbstractArray, dims::Int...)
    u = similar(x, size(x)..., dims...)
    f!(rng, u)
    return u
end

rsimilar(rng, f!, x::AbstractArray, dims::Int...) = _rsimilar(rng, f!, x, dims...)

randsimilar(rng::AbstractRNG, x::AbstractArray, dims::Int...) = rsimilar(rng, rand!, x, dims...)
randnsimilar(rng::AbstractRNG, x::AbstractArray, dims::Int...) = rsimilar(rng, randn!, x, dims...)

### Use below if we don't want to use in-place rand cores

#  function _rsimilar(rng::AbstractRNG, f::Function, x::AbstractArray, dims::Int...)
#      u = f(rng, eltype(x), size(x)..., dims...)
#      return u
#  end

#  rsimilar(rng, f, x::AbstractArray, dims::Int...) = _rsimilar(rng, f, x, dims...)

#  randsimilar(rng::AbstractRNG, x::AbstractArray, dims::Int...) = rsimilar(rng, rand, x, dims...)
#  randnsimilar(rng::AbstractRNG, x::AbstractArray, dims::Int...) = rsimilar(rng, randn, x, dims...)

### Distributions

using Distributions: Distributions, VariateForm, ValueSupport, Discrete, Continuous, Distribution, ContinuousMultivariateDistribution
import Distributions: logpdf, pdf, cdf, invlogcdf, ccdf, rand, mean, std, var, mode, minimum, maximum
import Distributions: Bernoulli, Normal

struct Batch <: VariateForm end
const BatchDistribution{S<:ValueSupport} = Distribution{Batch,S}
const DiscreteBatchDistribution = Distribution{Batch,Discrete}
const ContinuousBatchDistribution = Distribution{Batch,Continuous}
rand(bd::BatchDistribution, dims::Int...; kwargs...) = rand(GLOBAL_RNG, bd, dims...; kwargs...)

include("noise.jl")
export UniformNoise, GaussianNoise
include("normal.jl")
export Normal, NormalStd, BroadcastedNormalStd, NormalVar, BroadcastedNormalVar, NormalLogStd, BroadcastedNormalLogStd, NormalLogVar, BroadcastedNormalLogVar
include("gumbel.jl")
export GumbelSoftmax, GumbelSoftmax2D, GumbelBernoulli, GumbelBernoulliLogit
include("bernoulli.jl")
export BatchBernoulli, BatchBernoulliLogit
export Bernoulli, BernoulliLogit

export logpdf, pdf, cdf, invlogcdf, ccdf, rand, mean, std, var, mode, minimum, maximum
export logpdflogit, logpdfCoV, logrand, logitrand, logvar, kldiv

### X

include("ad.jl")

function __init__()
    Requires.@require Flux="587475ba-b771-5e3f-ad9e-33799f191a9c" begin
        import Flux

        Flux.adapt(::Type{Array}, x::UniformNoise{T,<:Any,S}) where {T,S} = UniformNoise(T, :cpu, x.size)
        Flux.adapt(::Type{Array}, x::GaussianNoise{T,<:Any,S}) where {T,S} = GaussianNoise(T, :cpu, x.size)

        for T in [
            GumbelSoftmax,
            GumbelBernoulli,
            GumbelBernoulliLogit,
            BatchBernoulli,
            BatchBernoulliLogit,
        ]
            @eval Flux.@functor $T
        end
    end

    Requires.@require CuArrays="3a865a2d-5b23-5a0f-bc46-62713ec82fae" begin
        import CuArrays

        # NOTE: The two functions below achive the following behaviour
        # - Pass `rng` if it's of type `CuArrays.CURAND.RNG`;
        # - Ignore `rng` and use the global of `CuArrays.CURAND` otherwise.
        # The motivation is to avoid scalar operations on GPUs, which is the case when
        # a CPU's RNG is used for inplace random number generation on GPUs.
        rsimilar(rng::CuArrays.CURAND.RNG, f!, x::CuArrays.CuArray, dims::Int...) = _rsimilar(rng, f!, x, dims...)
        rsimilar(::AbstractRNG, f!, x::CuArrays.CuArray, dims::Int...) = _rsimilar(CuArrays.CURAND.generator(), f!, x, dims...)

        ### Use below if we want rsimilar to be reproducible

        # TODO: add a global switch
        #  rsimilar(rng::AbstractRNG, f, x::CuArrays.CuArray, dims::Int...) = _rsimilar(rng, f, x, dims...) |> cu

        CuArrays.cu(x::UniformNoise) = UniformNoise(Float32, :gpu, x.size)
        CuArrays.cu(x::GaussianNoise) = GaussianNoise(Float32, :gpu, x.size)
    end
end

### Test

using Test: @test

"""
    test_stat(fstat, dist, n_samples, atol; samples)

Test the empirical statistic using samples against the true one.
"""
function test_stat(
    fstat::Function,
    dist::BatchDistribution,
    samples::AbstractArray,
    atol::AbstractFloat
)
    dim = length(size(samples))
    stat_est = dropdims(fstat(samples; dims=dim); dims=dim)
    stat_exact = fstat(dist)
    @test stat_est ≈ stat_exact atol=atol
    return samples
end

test_stat(
    fstat::Function,
    dist::BatchDistribution,
    n_samples::Int,
    atol::AbstractFloat
) = test_stat(fstat, dist, rand(dist, n_samples), atol)

end # module
