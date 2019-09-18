__precompile__()
module MLToolkit

greet() = print("Welcome to Kai's machine learning toolkit!")

# Package level imports all go here
import PyCall, PyPlot, Distributions, Reexport, Distributed, Tracker

# Pre-allocating Python bindings
const axes_grid1 = PyCall.PyNULL()
# Matplotlib and PyPlot
const mlp = PyCall.PyNULL()
const plt = PyCall.PyNULL()

###############################
# Constants that are exported #
###############################
const FT = Float64
export FT, mpl, plt

include("utility.jl")
export count_leadingzeros, turnoffgpu
include("special.jl")
export lbeta, beta, logit, logsumexp
include("transformations.jl")
export break_stick_ibp, break_logstick_ibp
include("Data/Data.jl")
Reexport.@reexport using .Data
include("scripting.jl")
export parse_args, flatten_dict, dict2namedtuple, merge_namedtuples, map_namedtuple, args_dict2str, isjupyter, @jupyter, @script, checknumerics, @checknumerics, sweepcmd, sweeprun, CombinedLogger
include("plotting.jl")
export make_two_y_axes_plot, plot_grayimg!, plot_actmat!, autoset_lim!, plot_contour!

include("distributions/distributions.jl")
include("MonteCarlo/MonteCarlo.jl")
Reexport.@reexport using .MonteCarlo
# include("neural/neural.jl")

include("test_util.jl")
export NUM_RANDTESTS, ATOL, ATOL_RAND, include_list_as_module

# Module init
function __init__()
    # Bind Python libraries
    copy!(axes_grid1, PyCall.pyimport("mpl_toolkits.axes_grid1"))
    copy!(mlp, PyPlot.matplotlib)
    copy!(plt, mlp.pyplot)
    # Ensure not using Type 3 fonts
    plt.rc("pdf", fonttype=42)
end

end # module
