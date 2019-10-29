module Neural

using ..MLToolkit: usegpu, FloatT
using Random: AbstractRNG, GLOBAL_RNG
using Flux

import Flux, Tracker, Distributions

### Tracker extensions

trackerparams(m) = m |> Flux.params |> Tracker.Params
trackergradient = Tracker.gradient

# https://github.com/FluxML/Flux.jl/blob/bdeb9c6d584668c7cef1ce71caf659d611c86d65/src/optimise/train.jl#L9-L18
function trackerapply!(opt, xs::Tracker.Params, Δs)
    for x in xs
        Δs[x] == nothing && continue
        x.data .-= Flux.Optimise.apply!(opt, Tracker.data(x), Tracker.data(Δs[x]))
    end
end

using Tracker: Params, losscheck, @interrupts, back!, Grads, tracker, extract_grad!

function Tracker.gradient(f, xs::Params; once=true)
    l = f()
    losscheck(l)
    @interrupts back!(l; once=once)
    gs = Grads()
    for x in xs
        gs[tracker(x)] = extract_grad!(x)
    end
    return gs
end

track_arr(x) = x isa AbstractArray ? Tracker.TrackedArray(x) : x
track(m) = Flux.fmap(track_arr, m)

# Avoid tracking some fields
function Flux.fmap(f::typeof(track_arr), m::BatchNorm; cache=IdDict())
    haskey(cache, m) && return cache[x]
    cache[m] = BatchNorm(m.λ, f(m.β), f(m.γ), m.μ, m.σ², m.ϵ, m.momentum)
end

export trackerparams, trackergradient, trackerapply!, track

### Flux extensions

nparams(m) = sum(prod.(size.(Flux.params(m))))

export nparams

###

abstract type AbstractNeuralModel end

include("architecture.jl")
export DenseNet, ConvNet
include("gen.jl")
export NeuralSampler, Discriminator, Projector

end # module
