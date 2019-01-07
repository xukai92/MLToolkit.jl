import Distributions: logpdf, pdf, rand, mode

include("distributions/displaced_poisson.jl")
export DisplacedPoisson
include("distributions/ibp.jl")
export IBP
include("distributions/normal.jl")
export BatchNormal
include("distributions/gumbel.jl")
export BatchGumbelSoftmax, BatchGumbelSoftmax2D, BatchGumbelBernoulli, BatchGumbelBernoulliLogit
include("distributions/bernoulli.jl")
export BatchBernoulli
include("distributions/beta.jl")
export BatchKumaraswamy, BatchBeta

export logpdf, pdf, logpdflogit, logrand, rand, logitrand, kl, mode
