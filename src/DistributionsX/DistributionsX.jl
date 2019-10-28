module DistributionsX

using ..MLToolkit: usegpu, FloatT
using Distributions: ContinuousMultivariateDistribution
using Random: AbstractRNG

import Random, NNlib, Tracker, Flux
import Distributions: logpdf, pdf, cdf, invlogcdf, ccdf, rand, mean, mode, minimum, maximum

function rsimilar(f!::Function, x::AbstractArray{FT}, n::Int) where {FT<:AbstractFloat}
    sz = n == 1 ? size(x) : (size(x)..., n)
    u = similar(Tracker.data(x), sz...)
    f!(u)
    return u
end

randsimilar(x, n::Int=1) = rsimilar(Random.rand!, x, n)
randnsimilar(x, n::Int=1) = rsimilar(Random.randn!, x, n)

include("noise.jl")
export UniformNoise, GaussianNoise
include("gumbel.jl")
export GumbelSoftmax#, GumbelSoftmax2D, GumbelBernoulli, GumbelBernoulliLogit

end # module
