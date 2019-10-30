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
function _rsimilar(rng::AbstractRNG, f!::Function, x::AbstractArray, n::Int)
    sz = n == 1 ? size(x) : (size(x)..., n)
    u = similar(x, sz...)
    f!(rng, u)
    return u
end

rsimilar(rng, f!, x::AbstractArray, n) = _rsimilar(rng, f!, x, n)

randsimilar(rng::AbstractRNG, x::AbstractArray, n::Int=1) = rsimilar(rng, rand!, x, n)
randnsimilar(rng::AbstractRNG, x::AbstractArray, n::Int=1) = rsimilar(rng, randn!, x, n)

### Distributions

using Distributions: VariateForm, ValueSupport, Discrete, Continuous, Distribution, ContinuousMultivariateDistribution
import Distributions: logpdf, pdf, cdf, invlogcdf, ccdf, rand, mean, var, mode, minimum, maximum

struct Batch <: VariateForm end
const BatchDistribution{S<:ValueSupport} = Distribution{Batch,S}
const DiscreteBatchDistribution = Distribution{Batch,Discrete}
const ContinuousBatchDistribution = Distribution{Batch,Continuous}
rand(bd::BatchDistribution, d::Int=1, dims::Int...; kwargs...) = rand(GLOBAL_RNG, bd, d, dims...; kwargs...)
function logpdf(d::BatchDistribution, x; kwargs...)
    if !(size(x) == size(d) || Base.front(size(x)) == size(d))
        throw(DimensionMismatch("size of input x $(size(x)) does not match size of distribution $(size(d))"))
    end
    return _logpdf(d, x; kwargs...)
end

include("noise.jl")
export UniformNoise, GaussianNoise
include("gumbel.jl")
export GumbelSoftmax, GumbelSoftmax2D, GumbelBernoulli, GumbelBernoulliLogit
include("bernoulli.jl")
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
    Bernoulli,
    BernoulliLogit,
]
    @eval Flux.@functor $T
end

end # module
