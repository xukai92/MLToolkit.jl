module Neural

using Random: AbstractRNG, GLOBAL_RNG
using Flux

import Flux, Zygote, Tracker, Distributions, BSON

### Tracker extensions

trackerparams(m) = m |> Flux.params |> Tracker.Params

Flux.gradient(f, ps::Tracker.Params; kwargs...) = Tracker.gradient(f, ps; kwargs...)
Flux.data(xs::Tracker.Params) = Tracker.data(xs)
Flux.data(xs::Tracker.TrackedArray) = Tracker.data(xs)

# https://github.com/FluxML/Flux.jl/blob/bdeb9c6d584668c7cef1ce71caf659d611c86d65/src/optimise/train.jl#L9-L18
function Flux.Optimise.update!(opt, xs::Tracker.Params, Δs)
    for x in xs
        Δs[x] == nothing && continue
        Tracker.update!(x, -Flux.Optimise.apply!(opt, Tracker.data(x), Tracker.data(Δs[x])))
    end
end

function Tracker.gradient(f, xs::Tracker.Params; once=true)
    l = f()
    Tracker.losscheck(l)
    Tracker.@interrupts Tracker.back!(l; once=once)
    gs = Tracker.Grads()
    for x in xs
        gs[Tracker.tracker(x)] = Tracker.extract_grad!(x)
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

export trackerparams, track

### Flux extensions

nparams(m) = sum(prod.(size.(Flux.params(m))))

export nparams

### Zygote extensions

function Zygote.pullback(f, p1::Flux.Params, p2::Flux.Params, prest::Flux.Params...)
    ps = (p1, p2, prest...)
    n = length(ps)
    cx = Zygote.Context()
    ys, back = Zygote._pullback(cx, f)
    @assert length(ys) == n
    caches = [copy(cx.cache) for i in 1:n]
    function back_i(Δ, i)
        cx.cache = caches[i]
        for p in ps[i]
          Zygote.cache(cx)[p] = nothing
        end
        back(Δ)
        Zygote.Grads(cx.cache)
    end
    ys, tuple((Δ -> back_i(Δ, i) for i in 1:n)...)
end

function Zygote.gradient(f, p1::Flux.Params, p2::Flux.Params, prest::Flux.Params...)
    ys, backs = Zygote.pullback(f, p1, p2, prest...)
    n = length(ys)
    makesens(i) = tuple(map(j -> i == j ? Zygote.sensitivity(ys[i]) : zero(ys[i]), 1:n)...)
    return tuple([backs[i](makesens(i)) for i in 1:n]...)
end

Zygote.@nograd Flux.gpu

# TODO: enable below if saving and loading params is buggy
# Flux.trainable(bn::BatchNorm) = (bn.β, bn.γ, bn.μ, bn.σ²)

###

abstract type AbstractNeuralModel end

include("trainable.jl")
export train!, saveparams, loadparams!

NeuralNet(Sin::Int, Sout::Int, args...) = DenseNet(Sin, Sout, args...)
NeuralNet(Sin::NTuple{3, Int}, Sout::Int, args...) = ConvNet(Sin, Sout, args...)
NeuralNet(Sin::Int, Sout::NTuple{3, Int}, args...) = ConvNet(Sin, Sout, args...)
include("architecture.jl")
export NeuralNet, DenseNet, ConvNet

export Neural

end # module
