module MLToolkit

export MLToolkit

greet() = print("Welcome to Kai's machine learning toolkit!")

# Package level imports all go here
import Distributions, Reexport, Requires, Tracker, Parameters, LinearAlgebra, Random

### Constants

const Float = Float32
const Double = Float64

export Float, Double

### Homeless functions

Base.sum(x::AbstractArray, drop::Val{:drop}; dims=:) = dropdims(sum(x; dims=dims); dims=dims)
Base.getproperty(ts::AbstractArray{<:NamedTuple}, k::Symbol) = getproperty.(ts, k)
Base.getindex(ts::AbstractArray{<:NamedTuple}, k::Symbol) = getindex.(ts, k)

# TODO: clean-up `special`
include("special.jl")

# TODO: move `transformations` somewhere
include("transformations.jl")
export break_stick_ibp, break_logstick_ibp

include("Plots/Plots.jl")
Reexport.@reexport using .Plots
include("Datasets/Datasets.jl")
Reexport.@reexport using .Datasets
include("Scripting/Scripting.jl")
Reexport.@reexport using .Scripting

# TODO: merge `distributions` and `DistributionsX`
include("distributions/distributions.jl")
include("DistributionsX/DistributionsX.jl")
Reexport.@reexport using .DistributionsX
include("MonteCarlo/MonteCarlo.jl")
Reexport.@reexport using .MonteCarlo
# TODO: merge `neural.deprecated` (which contains codes for RAVE) and `Neural`
# include("neural.deprecated/neural.jl")

function __init__()
    Requires.@require CuArrays="3a865a2d-5b23-5a0f-bc46-62713ec82fae" begin
        include("gpu.jl")
        export seed!
    end

    Requires.@require Flux="587475ba-b771-5e3f-ad9e-33799f191a9c" begin
        include("Neural/Neural.jl")
        Reexport.@reexport using .Neural
    end
end

end # module
