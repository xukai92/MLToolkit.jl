# TODO: make all local vs global type consistent

import Distributions: logpdf, pdf, cdf, invlogcdf, ccdf, rand, mean, mode, minimum, maximum

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
