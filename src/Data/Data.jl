module Data

using ..MLToolkit: FT, AT
export FT, AT

include("mnist.jl")
export load_mnist
include("loader.jl")
export BatchDataLoader
include("visualisation.jl")
export make_imggrid

end # module
