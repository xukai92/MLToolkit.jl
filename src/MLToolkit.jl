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

include("special.jl")
export lbeta, beta
include("activations.jl")
export softplus, leaky_relu
include("transformations.jl")
export break_stick_ibp, break_logstick_ibp
include("data.jl")
export load_mnist

include("distributions.jl")

end
