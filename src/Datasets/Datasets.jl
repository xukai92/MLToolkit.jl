module Datasets

using Random: MersenneTwister

abstract type AbstractDataset end

n_display(d::AbstractDataset) = throw(MethodError(n_display, d))

include("ring.jl")
export RingDataset
include("gaussian.jl")
export GaussianDataset
include("mnist.jl")
export MNISTDataset
include("features.jl")
export FeatureDataset, get_features_griffiths2011, get_features_xu2019

export Datasets, n_display

end # module
