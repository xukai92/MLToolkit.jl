module MLToolkit

# Package level imports all go here
import AutoGrad, Knet, PyPlot, Distributions, Reexport
const plt = PyPlot

# Constants
const FT = Float64
const AT = Knet.gpu() != -1 ? Knet.KnetArray : Array    # use GPU whenever possible

# For test purposes
const NUM_RANDTESTS = 5
const ATOL = FT == Float64 ? 1e-6 : 1e-4
const ATOL_RAND = FT == Float64 ? 2e-2 : 5e-1

export NUM_RANDTESTS, ATOL, ATOL_RAND, FT, AT, plt

greet() = print("Welcome to Kai's machine learning toolkit!")

include("special.jl")
export lbeta, beta, logit
include("transformations.jl")
export break_stick_ibp, break_logstick_ibp
include("Data/Data.jl")
Reexport.@reexport using .Data
include("scripting.jl")
export parse_args
include("plotting.jl")
export plot_two_y_axes

include("distributions.jl")
include("MonteCarlo/MonteCarlo.jl")
Reexport.@reexport using .MonteCarlo
include("neural.jl")

end # module
