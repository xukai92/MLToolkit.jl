module DistributionsX

using Distributions: ContinuousMultivariateDistribution
using Random: AbstractRNG

import Distributions: logpdf, pdf, cdf, invlogcdf, ccdf, rand, mean, mode, minimum, maximum

include("std.jl")
export DiagUniform, DiagStdNormal

end # module
