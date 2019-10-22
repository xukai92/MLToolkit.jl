module Data

using ..MLToolkit: FT
export FT

# include("mnist.jl")
# export load_mnist
include("loader.jl")
export BatchDataLoader, shuffle!
include("features.jl")
export get_features_griffiths2011indian, get_features_large

end # module
