module DistributionsX

using Random: AbstractRNG, GLOBAL_RNG, rand!, randn!
import StatsFuns, NNlib

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

### Distributions

using Distributions: Distributions, VariateForm, ValueSupport, Discrete, Continuous, Distribution, ContinuousMultivariateDistribution
import Distributions: logpdf, pdf, cdf, invlogcdf, ccdf, rand, mean, var, mode, minimum, maximum
import Distributions: Bernoulli

struct Batch <: VariateForm end
const BatchDistribution{S<:ValueSupport} = Distribution{Batch,S}
const DiscreteBatchDistribution = Distribution{Batch,Discrete}
const ContinuousBatchDistribution = Distribution{Batch,Continuous}
rand(bd::BatchDistribution, dims::Int...; kwargs...) = rand(GLOBAL_RNG, bd, dims...; kwargs...)

include("noise.jl")
export UniformNoise, GaussianNoise
include("gumbel.jl")
export GumbelSoftmax, GumbelSoftmax2D, GumbelBernoulli, GumbelBernoulliLogit
include("bernoulli.jl")
export BatchBernoulli, BatchBernoulliLogit
export Bernoulli, BernoulliLogit

export logpdf, pdf, cdf, invlogcdf, ccdf, rand, mean, var, mode, minimum, maximum
export logpdflogit, logpdfCoV, logrand, logitrand, kldiv

### X

using Tracker, Flux

include("ad.jl")

Flux.use_cuda && include("gpu.jl")

for T in [
    GumbelSoftmax,
    GumbelBernoulli,
    GumbelBernoulliLogit,
    BatchBernoulli,
    BatchBernoulliLogit,
]
    @eval Flux.@functor $T
end

### Test

using Test: @test

"""
    test_stat(fstat, dist, n_samples, atol; samples)

Test the emperical statistic using samples against the true one.
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
