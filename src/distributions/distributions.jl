# TODO: make all local vs global type consistent

import Distributions: logpdf, pdf, cdf, invlogcdf, ccdf, rand, mean, mode, minimum, maximum

# randarr(sz) = Knet.rand!(AT{FT,length(sz)}(undef, sz...))
# randnarr(sz) = Knet.randn!(AT{FT,length(sz)}(undef, sz...))
# function randsimilar(arr, n::Int=1)
#     T = typeof(AutoGrad.value(arr))
#     sz = n == 1 ? size(arr) : (size(arr, 1), n)
#     return Knet.rand!(T(undef, sz...))
# end
# function randnsimilar(arr, n::Int=1)
#     T = typeof(AutoGrad.value(arr))
#     sz = n == 1 ? size(arr) : (size(arr, 1), n)
#     return Knet.randn!(T(undef, sz...))
# end

include("displaced_poisson.jl")
export DisplacedPoisson
include("ibp.jl")
export IBP
include("normal.jl")
export BatchNormal, BatchNormalLogVar
# include("bernoulli.jl")
# export BatchBernoulli, BatchBernoulliLogit
include("beta.jl")
export BatchKumaraswamy, BatchBeta
include("npd.jl")
export LogitNPD, getlogitρ, getρ, getlogρ

export logpdf, pdf, cdf, invlogcdf, ccdf, rand, mean, mode, minimum, maximum
export logpdflogit, logpdfCoV, logrand, logitrand, kldiv
