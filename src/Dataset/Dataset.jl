module Dataset

using Random: AbstractRNG
import Distributions: rand, logpdf
using Distributions: ContinuousMultivariateDistribution, MixtureModel, MvNormal

include("ring.jl")
export RingDataset

include("features.jl")
export FeatureDataset, get_features_griffiths2011, get_features_xu2019

export rand, logpdf

end # module
