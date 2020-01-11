module MLToolkit

const MLT = MLToolkit
export MLT

greet() = print("Welcome to Kai's machine learning toolkit!")

# Package level imports all go here
import Distributions, Reexport, Distributed, Flux, Tracker, Parameters, LinearAlgebra, Random

### Constants

const FT = Float64  # TODO: remove this
const FloatT = Ref(Float32)
const usegpu = Ref(true)

export FloatT

### Homeless functions

include("utility.jl")
if Flux.use_cuda[]
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
Reexport.@reexport using .Datasets
include("Scripting/Scripting.jl")
Reexport.@reexport using .Scripting

# TODO: merge `distributions` and `DistributionsX`
include("distributions/distributions.jl")
include("DistributionsX/DistributionsX.jl")
Reexport.@reexport using .DistributionsX
include("MonteCarlo/MonteCarlo.jl")
Reexport.@reexport using .MonteCarlo
# include("neural/neural.jl")
include("Neural/Neural.jl")
Reexport.@reexport using .Neural

end # module
