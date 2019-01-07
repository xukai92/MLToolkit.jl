import AutoGrad: grad
import Knet: update!

using AutoGrad: Tape

abstract type AbstractTrainable end

"""
Initialise optimizer for each trainable parameters.
"""
function initoptim!(model::AbstractTrainable, otype; args...)
    for param in params(model)
        param.opt = otype(; args...)
    end
end

"""
Get the gradient dictionary for `params` from the result tape `x`.
"""
function grad(x::Tape, params::Array)
    return Dict(param => grad(x, param) for param in params)
end

"""
Get the gradient dictionary for all parameters of `model` from the result tape `x`.
"""
grad(x::Tape, model::AbstractTrainable) = grad(x, params(model))

"""
Update all parameters in `params` using the gradient dict `d`.
"""
function update!(params::Array, g::Dict)
    for param in params
        update!(value(param), g[param], param.opt)
    end
end

"""
Update all parameters of `model` using the gradient dict `d`.
"""
update!(model::AbstractTrainable, g::Dict) = update!(params(model), g)

include("neural/layers.jl")
export Dense, Chain

export initoptim!, grad, update!, AbstractTrainable
