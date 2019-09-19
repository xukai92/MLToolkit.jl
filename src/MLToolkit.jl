__precompile__()
module MLToolkit

greet() = print("Welcome to Kai's machine learning toolkit!")

# Package level imports all go here
import PyCall, PyPlot, Distributions, Reexport, Distributed, Tracker, Requires, Logging, TensorBoardLogger, Images

# Pre-allocating Python bindings
const axes_grid1 = PyCall.PyNULL()
const plt_agg = PyCall.PyNULL()
# Matplotlib and PyPlot
const mpl = PyCall.PyNULL()
const plt = PyCall.PyNULL()

###############################
# Constants that are exported #
###############################
const FT = Float64
export FT, mpl, plt

######################
# Homeless functions #
######################
nparams(m) = sum(prod.(size.(Flux.params(m))))
export nparams

include("utility.jl")
export count_leadingzeros, turnoffgpu
include("special.jl")
export lbeta, beta, logit, logsumexp
include("transformations.jl")
export break_stick_ibp, break_logstick_ibp
include("Data/Data.jl")
Reexport.@reexport using .Data
include("scripting.jl")
export DATETIME_FMT, find_latest_dir, parse_args, flatten_dict, dict2namedtuple, merge_namedtuples, map_namedtuple, args_dict2str, isjupyter, @jupyter, @script, @tb, checknumerics, @checknumerics, sweepcmd, sweeprun, CombinedLogger
include("plotting.jl")
export make_two_y_axes_plot, plot_grayimg!, plot_actmat!, autoset_lim!, plot_contour!, plot_pdf!

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
    copy!(mpl, PyPlot.matplotlib)
    copy!(plt, mpl.pyplot)
    copy!(plt_agg, mpl.backends.backend_agg)
    # Ensure not using Type 3 fonts
    plt.rc("pdf", fonttype=42)
    # GPU support
    Requires.@require CuArrays="3a865a2d-5b23-5a0f-bc46-62713ec82fae" include("gpu.jl")
end

end # module
