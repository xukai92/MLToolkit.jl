module Data

using ..MLToolkit: FT, AT
export FT, AT

include("mnist.jl")
export load_mnist
include("loader.jl")
export BatchDataLoader, shuffle!
include("visualisation.jl")
export make_imggrid
include("features.jl")
export get_features_griffiths2011indian, get_features_large

end # module
