module MLToolkit

greet() = print("Welcome to Kai's machine learning toolkit!")

# Package level imports all go here
import AutoGrad, Knet, PyPlot, Distributions, Reexport, Distributed

# Constants
const FT = Float32
const AT = Knet.gpu() != -1 ? Knet.KnetArray : Array    # use GPU whenever possible
const plt = PyPlot
export FT, AT, plt

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

include("distributions/distributions.jl")
include("MonteCarlo/MonteCarlo.jl")
Reexport.@reexport using .MonteCarlo
include("neural/neural.jl")

include("test_util.jl")
export NUM_RANDTESTS, ATOL, ATOL_RAND, include_list_as_module

end # module
