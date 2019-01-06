module MLToolkit

using Knet: gpu, KnetArray

const NUM_RANDTESTS = 5
const ATOL = 1e-6
const ATOL_RAND = 2e-2
const FT = Float64
# Use GPU whenever possible
const AT = gpu() != -1 ? KnetArray : Array
export NUM_RANDTESTS, ATOL, ATOL_RAND, FT, AT

greet() = print("Welcome to Kai's machine learning toolkit!")

include("activations.jl")
include("distributions.jl")
include("transformations.jl")

end
