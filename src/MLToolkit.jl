__precompile__()
module MLToolkit

greet() = print("Welcome to Kai's machine learning toolkit!")

# Module init
function __init__()
    # Bind Python libraries
    copy!(axes_grid1, PyCall.pyimport("mpl_toolkits.axes_grid1"))
end

# Package level imports all go here
import AutoGrad, Knet, PyCall, PyPlot, Distributions, Reexport, Distributed

# Pre-allocating Python bindings
const axes_grid1 = PyCall.PyNULL()

# Constants that are exported
const FT = Float32
const AT = Knet.gpu() != -1 ? Knet.KnetArray : Array    # use GPU whenever possible
const plt = PyPlot
export FT, AT, plt

include("utility.jl")
export count_leadingzeros
include("special.jl")
export lbeta, beta, logit
include("transformations.jl")
export break_stick_ibp, break_logstick_ibp
include("Data/Data.jl")
Reexport.@reexport using .Data
include("scripting.jl")
export parse_args, flatten_dict, @jupyter, @script
include("plotting.jl")
export make_two_y_axes_plot, plot_grayimg, plot_actmat

include("distributions/distributions.jl")
include("MonteCarlo/MonteCarlo.jl")
Reexport.@reexport using .MonteCarlo
include("neural/neural.jl")

include("test_util.jl")
export NUM_RANDTESTS, ATOL, ATOL_RAND, include_list_as_module

end # module
