module MLToolkit

export MLToolkit

greet() = print("Welcome to Kai's machine learning toolkit!")

using ArgCheck
import Distributions, DistributionsAD, Reexport, Requires, Tracker, Parameters, LinearAlgebra, Random

### Constants

const Float = Float32
const Double = Float64

export Float, Double

### Polluting functions

Base.sum(x::AbstractArray, drop::Val{:drop}; dims=:) = dropdims(sum(x; dims=dims); dims=dims)
Base.getproperty(ts::AbstractArray{<:NamedTuple}, k::Symbol) = getproperty.(ts, k)
Base.getindex(ts::AbstractArray{<:NamedTuple}, k::Symbol) = getindex.(ts, k)

Base.getindex(arr::AbstractArray{<:Any,1}, ::Colon, i) = arr[i]
Base.getindex(arr::AbstractArray{<:Any,3}, ::Colon, i) = arr[:,:,i]
Base.getindex(arr::AbstractArray{<:Any,4}, ::Colon, i) = arr[:,:,:,i]

function Random.shuffle(nt::NamedTuple{K, V}) where {K, V<:Tuple{Vararg{<:AbstractArray}}}
    ns = [last(size(v)) for v in values(nt)]
    n = first(ns)
    @argcheck all(n .== ns)
    idcs = Random.shuffle(1:n)
    return nt[:,idcs]
end

function Base.getindex(nt::NamedTuple{K, V}, ::Colon, i) where {K, V<:Tuple{Vararg{<:AbstractArray}}}
    v = tuple([v[:, i] for v in values(nt)]...)
    return NamedTuple{K, typeof(v)}(v)
end

Base.:(+)(nt1::T, nt2::T) where {T<:NamedTuple} = T(tuple((values(nt1) .+ values(nt2))...))

function Base.union(nt1::NamedTuple{S1, T1}, nt2::NamedTuple{S2, T2}) where {S1, T1, S2, T2}
    return NamedTuple{(S1..., S2...), Tuple{T1.types..., T2.types...}}((nt1..., nt2...))
end

# TODO: clean-up `special`
include("special.jl")

# TODO: move `transformations` somewhere
include("transformations.jl")
export break_stick_ibp, break_logstick_ibp

include("PlotsX/PlotsX.jl")
Reexport.@reexport using .PlotsX
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
# TODO: merge `neural.deprecated` (which contains codes for RAVE) and `Neural`; variational layers are also important
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
