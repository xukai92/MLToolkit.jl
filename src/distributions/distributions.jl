# TODO: make all local vs global type consistent

import Distributions: logpdf, pdf, cdf, invlogcdf, ccdf, rand, mean, mode, minimum, maximum

randarr(sz, T1=AT, T2=FT) = Knet.rand!(T1{T2,length(sz)}(undef, sz...))
function randsimilar(arr, n::Int=1)
    T = isa(arr, AutoGrad.Result) ? typeof(AutoGrad.value(arr)) : typeof(arr)
    sz = n == 1 ? size(arr) : (size(arr, 1), n)
    return Knet.rand!(T(undef, sz...))
end
randnarr(sz, T1, T2) = Knet.randn!(T1{T2,length(sz)}(undef, sz...))
function randnsimilar(arr, n::Int=1)
    T = isa(arr, AutoGrad.Result) ? typeof(AutoGrad.value(arr)) : typeof(arr)
    sz = n == 1 ? size(arr) : (size(arr, 1), n)
    return Knet.randn!(T(undef, sz...))
end

include("displaced_poisson.jl")
export DisplacedPoisson
include("ibp.jl")
export IBP
include("normal.jl")
export BatchNormal, BatchNormalLogVar
include("gumbel.jl")
export BatchGumbelSoftmax, BatchGumbelSoftmax2D, BatchGumbelBernoulli, BatchGumbelBernoulliLogit
include("bernoulli.jl")
export BatchBernoulli, BatchBernoulliLogit
include("beta.jl")
export BatchKumaraswamy, BatchBeta
include("npd.jl")
export LogitNPD, getlogitρ, getρ, getlogρ

export logpdf, pdf, cdf, invlogcdf, ccdf, rand, mean, mode, minimum, maximum
export logpdflogit, logpdfCoV, logrand, logitrand, kldiv
