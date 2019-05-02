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
Knet.gpu(false) # deactivate the GPU Knet activated at startup;
                # activation of GPU should be explicit

# Pre-allocating Python bindings
const axes_grid1 = PyCall.PyNULL()

# Constants that are exported
const FT = Float64
# Use GPU whenever possible
const AT = try
    # Check if the command `nvidia-smi` exists. If yes we use GPU.
    read(`command -v nvidia-smi`, String)
    Knet.KnetArray
catch
    Array
end
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
export parse_args, flatten_dict, isjupyter, @jupyter, @script, checknumerics, @checknumerics, sweepcmd, sweeprun, CombinedLogger
include("plotting.jl")
export make_two_y_axes_plot, plot_grayimg, plot_actmat

include("distributions/distributions.jl")
include("MonteCarlo/MonteCarlo.jl")
Reexport.@reexport using .MonteCarlo
include("neural/neural.jl")

include("test_util.jl")
export NUM_RANDTESTS, ATOL, ATOL_RAND, include_list_as_module

end # module
