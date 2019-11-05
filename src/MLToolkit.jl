module MLToolkit

const MLT = MLToolkit
export MLT

greet() = print("Welcome to Kai's machine learning toolkit!")

# Package level imports all go here
import PyCall, PyPlot, Distributions, Reexport, Distributed, Flux, Tracker, PGFPlots, Parameters, LinearAlgebra, Random

### Constants

# Pre-allocating Python bindings
const axes_grid1 = PyCall.PyNULL()
const plt_agg = PyCall.PyNULL()
# Matplotlib and PyPlot
const mpl = PyCall.PyNULL()
const plt = PyCall.PyNULL()

const FT = Float64  # TODO: remove this
const FloatT = Ref(Float32)
const usegpu = Ref(true)

export mpl, plt, FloatT

### Homeless functions

include("utility.jl")
if Flux.use_cuda
    include("gpu.jl")
    function seed!(s::Int)
        Random.seed!(s)
        Flux.CuArrays.CURAND.seed!(s)
    end
else
    function seed!(s::Int)
        Random.seed!(s)
    end
end
export count_leadingzeros, include_list_as_module, seed!
include("special.jl")
include("transformations.jl")
export break_stick_ibp, break_logstick_ibp
include("Data/Data.jl")
Reexport.@reexport using .Data
include("Scripting/Scripting.jl")
Reexport.@reexport using .Scripting
include("plotting.jl")
export TwoYAxesLines, GrayImages, make_imggrid, plot, plot!, save, plot_actmat!, autoset_lim!, ContourFunction

# TODO: merge `distributions` and `DistributionsX`
include("distributions/distributions.jl")
include("DistributionsX/DistributionsX.jl")
Reexport.@reexport using .DistributionsX
include("MonteCarlo/MonteCarlo.jl")
Reexport.@reexport using .MonteCarlo
# include("neural/neural.jl")
include("Neural/Neural.jl")
Reexport.@reexport using .Neural

### Module init

function __init__()
    # Do not show figures automatically in IJulia
    PyPlot.isjulia_display[] = false
    # Bind Python libraries
    copy!(axes_grid1, PyCall.pyimport("mpl_toolkits.axes_grid1"))
    copy!(mpl, PyPlot.matplotlib)
    copy!(plt, mpl.pyplot)
    copy!(plt_agg, mpl.backends.backend_agg)
    # Ensure not using Type 3 fonts
    plt.rc("pdf", fonttype=42)
end

end # module
