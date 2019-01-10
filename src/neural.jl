import AutoGrad: grad
import Knet: update!

abstract type AbstractTrainable end
abstract type AbstractLayer <: AbstractTrainable end
abstract type StaticLayer <: AbstractLayer end
abstract type StochasticLayer <: AbstractLayer end

"""
Initialise optimizer for each trainable parameters.
"""
function initoptim!(model::AbstractTrainable, otype; args...)
    for p in Knet.params(model)
        p.opt = otype(; args...)
    end
end

"""
Get the gradient dictionary for `ps` from the result tape `x`.
"""
function grad(x::AutoGrad.Tape, ps::Array)
    return Dict(p => grad(x, p) for p in ps)
end

"""
Get the gradient dictionary for all parameters of `model` from the result tape `x`.
"""
grad(x::AutoGrad.Tape, model::AbstractTrainable) = grad(x, Knet.params(model))

"""
Update all parameters in `ps` using the gradient dict `d`.
"""
function update!(ps::Array, g::Dict)
    for p in ps
        update!(AutoGrad.value(p), g[p], p.opt)
    end
end

"""
Update all parameters of `model` using the gradient dict `d`.
"""
update!(model::AbstractTrainable, g::Dict) = update!(Knet.params(model), g)

function numparams(model::AbstractTrainable)
    n = 0
    for p in Knet.params(model)
        n += prod(size(p))
    end
    return n
end

include("neural/activations.jl")
export softplus, leaky_relu
include("neural/layers.jl")
export Dense, DynamicIn, DynamicOut
export GaussianDense, GaussianDynamicIn, GaussianDynamicOut
export GaussianLogVarDense, GaussianLogVarDynamicIn, GaussianLogVarDynamicOut
export BernoulliDense, BernoulliDynamicIn, BernoulliDynamicOut
export BernoulliLogitDense, BernoulliLogitDynamicIn, BernoulliLogitDynamicOut

"""
Chaining multiple layers.

NOTE: only chainning layers are allowed but not models. As models are assumed to output loss when being called.
"""
struct Chain <: AbstractLayer
    layers::Tuple
    function Chain(layers::Tuple)
        n = length(layers)
        for i = 1:n-1
            @assert layers[i] isa StaticLayer "The layers in middle should be `StaticLayer`."
        end
        @assert layers[i] isa AbstractLayer "The last layer should be a `AbstractLayer`"
        return new(layers)
    end
end
Chain(layers::AbstractLayer...) = Chain(layers)
Chain(layers) = Chain(layers...)

"""
Run chained layers.
"""
function (c::Chain)(x)
    for l in c.layers
        x = l(x)
    end
    return x
end

"""
Run chained layers with `args...` applied to the last one.
"""
function (c::Chain)(x, args...)
    n = length(c.layers)
    for i = 1:n-1
        x = c.layers[i](x)
    end
    return c.layers[n](x, args...)
end

export initoptim!, grad, update!, numparams
export AbstractTrainable, AbstractLayer, StaticLayer, StochasticLayer, Chain
