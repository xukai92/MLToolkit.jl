module MLToolkit

const MLT = MLToolkit
export MLT

greet() = print("Welcome to Kai's machine learning toolkit!")

# Package level imports all go here
import Distributions, Reexport, Flux, Tracker, Parameters, LinearAlgebra, Random

### Constants

const FT = Float64  # TODO: remove this
const FloatT = Ref(Float32)
const usegpu = Ref(true)

export FloatT

### Homeless functions

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
export seed!

Base.sum(x::AbstractArray, drop::Val{:drop}; dims=:) = dropdims(sum(x; dims=dims); dims=dims)

function Base.getproperty(ts::AbstractArray{<:NamedTuple}, k::Symbol)
    return map(t -> getproperty(t, k), ts)
end

include("special.jl")
include("transformations.jl")
export break_stick_ibp, break_logstick_ibp
include("Datasets/Datasets.jl")
Reexport.@reexport using .Datasets
include("Plots/Plots.jl")
Reexport.@reexport using .Plots
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
