module Data

using Random: AbstractRNG
import Distributions: rand, logpdf
using Distributions: ContinuousMultivariateDistribution, MixtureModel, MvNormal

include("ring.jl")
export RingDataset

include("features.jl")
export LatentFeatureDataset, get_features_griffiths2011indian, get_features_large

export rand, logpdf

end # module
