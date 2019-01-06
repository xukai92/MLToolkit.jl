import Distributions: logpdf, pdf, rand, mode

include("distributions/displaced_poisson.jl")
export DisplacedPoisson

include("distributions/ibp.jl")
export IBP

include("distributions/normal.jl")
export UnivariateNormal, DiagonalNormal, DenseNormal

export logpdf, pdf, rand, kl, mode
