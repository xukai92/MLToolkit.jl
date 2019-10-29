module DistributionsX

using ..MLToolkit: usegpu, FloatT
using Distributions: ContinuousMultivariateDistribution
using Random: AbstractRNG

import Requires, Random, NNlib, Tracker, Flux
import Distributions: logpdf, pdf, cdf, invlogcdf, ccdf, rand, mean, mode, minimum, maximum

### MLE

Base.eltype(x::Tracker.TrackedArray) = eltype(Tracker.data(x))
Base.eps(x::AbstractArray) = eps(eltype(x))

"""
    rsimilar(rng, f!, x, n)

Generate random numbers in a container similar to `x`.
The returned variable will have the same shape of `x` if `n == 1` and
will extend an extra dimension with size `n` if `n != 1`.
"""
function rsimilar(rng::AbstractRNG, f!::Function, x::AbstractArray, n::Int)
    sz = n == 1 ? size(x) : (size(x)..., n)
    u = similar(x, sz...)
    f!(rng, u)
    return u
end

function rsimilar(rng::AbstractRNG, f!::Function, x::Tracker.TrackedArray, n::Int)
    return rsimilar(rng, f!, Tracker.data(x), n)
end

randsimilar(rng::AbstractRNG, x, n::Int=1) = rsimilar(rng, Random.rand!, x, n)
randnsimilar(rng::AbstractRNG, x, n::Int=1) = rsimilar(rng, Random.randn!, x, n)

Requires.@require CuArrays="3a865a2d-5b23-5a0f-bc46-62713ec82fae" include("gpu.jl")

### Distributions

include("noise.jl")
export UniformNoise, GaussianNoise
include("gumbel.jl")
export GumbelSoftmax#, GumbelSoftmax2D, GumbelBernoulli, GumbelBernoulliLogit

end # module
