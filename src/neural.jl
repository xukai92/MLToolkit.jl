import AutoGrad: grad
import Knet: update!

using AutoGrad: Tape, value
using Knet: params

abstract type AbstractTrainable end

"""
Initialise optimizer for each trainable parameters.
"""
function initoptim!(model::AbstractTrainable, otype; args...)
    for p in params(model)
        p.opt = otype(; args...)
    end
end

"""
Get the gradient dictionary for `ps` from the result tape `x`.
"""
function grad(x::Tape, ps::Array)
    return Dict(p => grad(x, p) for p in ps)
end

"""
Get the gradient dictionary for all parameters of `model` from the result tape `x`.
"""
grad(x::Tape, model::AbstractTrainable) = grad(x, params(model))

"""
Update all parameters in `params` using the gradient dict `d`.
"""
function update!(ps::Array, g::Dict)
    for p in ps
        update!(value(p), g[p], p.opt)
    end
end

"""
Update all parameters of `model` using the gradient dict `d`.
"""
update!(model::AbstractTrainable, g::Dict) = update!(params(model), g)

include("neural/layers.jl")
export Dense, Chain

export initoptim!, grad, update!, AbstractTrainable
