import AutoGrad: grad
import Knet: update!, train!

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
        @assert p.opt != nothing "$p has no optimizer set up"
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

"""
Neural models.

The difference between a layer and a model is that for models, loss are defined.

All `NeuralModel` should be compatible with the following signature.
```julia
Base.show(nm::NeuralModel)
loss(nm::NeuralModel, data)
eval(nm::NeuralModel, data)
```
"""
abstract type NeuralModel <: AbstractTrainable end
Base.show(nm::NeuralModel) = error("Method not implemented")
loss(nm::NeuralModel, data) = error("Method not implemented")
# Callback function - if no specific metric implemented, re-use the loss function
eval(nm::NeuralModel, data) = loss(nm, data)

"""
Train the model on a dataset.

For each batch in the dataset, do:
1. Compute the gradient of loss;
2. Update model parameters using the optimizers set for each parameters.

It returns batch averaged loss in the end.
"""
function train!(model::NeuralModel, dataloader)
    loss_list = []
    for data_batch in dataloader
        losstape = Knet.@diff loss(model, data_batch)
        graddict = grad(losstape, model)
        update!(model, graddict)
        push!(loss_list, Knet.value(losstape))
    end
    return mean(loss_list)
end

"""
Evaluate the model on a dataset.

It returns batch averaged metric value in the end.
"""
function evaluate(model::NeuralModel, dataloader)
    loss_list = []
    for data_batch in dataloader
        push!(loss_list, eval(model, data_batch))
    end
    return mean(loss_list)
end

include("neural/activations.jl")
export softplus, leaky_relu
include("neural/layers.jl")
export Dense, DynamicIn, DynamicOut
export GaussianDense, GaussianDynamicIn, GaussianDynamicOut
export GaussianLogVarDense, GaussianLogVarDynamicIn, GaussianLogVarDynamicOut
export BernoulliDense, BernoulliDynamicIn, BernoulliDynamicOut
export BernoulliLogitDense, BernoulliLogitDynamicIn, BernoulliLogitDynamicOut
export Chain

export initoptim!, grad, update!, numparams
export AbstractTrainable, AbstractLayer, StaticLayer, StochasticLayer
export NeuralModel, loss, eval, train!, evaluate
