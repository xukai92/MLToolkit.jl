import AutoGrad: grad
import Knet: update!

abstract type AbstractTrainable end

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
export Dense, GaussianNode, GaussianLogVarNode, BernoulliNode, BernoulliLogitNode

"""
Chaining multiple layers.
"""
struct Chain <: AbstractTrainable
    layers
end

function (c::Chain)(x)
    for l in c.layers
        x = l(x)
    end
    return x
end

export initoptim!, grad, update!, numparams, AbstractTrainable, Chain
