module MLToolkit

using Knet: gpu, KnetArray

const FT = Float64
# Use GPU whenever possible
const AT = gpu() != -1 ? KnetArray : Array
const NUM_RANDTESTS = 5
const ATOL = FT == Float64 ? 1e-6 : 1e-4
const ATOL_RAND = FT == Float64 ? 2e-2 : 5e-1

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
include("neural.jl")

end
