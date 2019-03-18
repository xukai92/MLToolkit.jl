import Distributions: logpdf, pdf, cdf, ccdf, rand, mean, mode, minimum, maximum

include("distributions/displaced_poisson.jl")
export DisplacedPoisson
include("distributions/ibp.jl")
export IBP
include("distributions/normal.jl")
export BatchNormal, BatchNormalLogVar
include("distributions/gumbel.jl")
export BatchGumbelSoftmax, BatchGumbelSoftmax2D, BatchGumbelBernoulli, BatchGumbelBernoulliLogit
include("distributions/bernoulli.jl")
export BatchBernoulli, BatchBernoulliLogit
include("distributions/beta.jl")
export BatchKumaraswamy, BatchBeta
include("distributions/npd.jl")
export LogitNPD, getlogitρ, getρ, getlogρ

export logpdf, pdf, cdf, ccdf, rand, mean, mode, minimum, maximum
export logpdflogit, logrand, logitrand, kldiv
