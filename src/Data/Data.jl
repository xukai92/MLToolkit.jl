module Data

using ..MLToolkit: FT, AT

include("mnist.jl")
include("loader.jl")
include("visualisation.jl")

export load_mnist, make_imggrid, BatchDataLoader,
       FT, AT

end # module
