import Distributions: pdf, rand, mode

include("distributions/displaced_poisson.jl")
export DisplacedPoisson

include("distributions/ibp.jl")
export IBP

export pdf, rand, mode
