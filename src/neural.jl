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
Update all parameters in `ps` by back-propgating `losstape`.
"""
function update!(ps::Array, losstape::AutoGrad.Tape)
    for p in ps
        g = AutoGrad.grad(losstape, p)
        @assert p.opt != nothing "$p has no optimizer set up"
        update!(AutoGrad.value(p), g, p.opt)
    end
end

"""
Update all parameters of `model` by back-propgating `losstape`.
"""
update!(model::AbstractTrainable, losstape::AutoGrad.Tape) = update!(Knet.params(model), losstape)

"""
Compute the number of parameters.
"""
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
(nm::NeuralModel)(data, training::Val{true})    # loss
(nm::NeuralModel)(data, training::Val{false})   # evaluate metric
```
"""
abstract type NeuralModel <: AbstractTrainable end
Base.show(nm::NeuralModel) = error("Method not implemented")

"""
Train the model on a dataset.

For each batch in the dataset, do:
1. Compute the gradient of loss;
2. Update model parameters using the optimizers set for each parameters.

It returns batch averaged loss in the end.
"""
function train!(model::NeuralModel, dataloader; kargs...)
    loss_list = []
    for data_batch in dataloader
        losstape = Knet.@diff model(data_batch, Val(:true); kargs...)
        update!(model, losstape)
        push!(loss_list, Knet.value(losstape))
    end
    return mean(loss_list)
end

"""
Evaluate the model on a dataset.

It returns batch averaged metric value in the end.
"""
function evaluate(model::NeuralModel, dataloader; kargs...)
    loss_list = []
    for data_batch in dataloader
        push!(loss_list, model(data_batch, Val(:false); kargs...)[1])
    end
    return mean(loss_list)
end

include("neural/activations.jl")
export softplus, leaky_relu
include("neural/layers.jl")
export Dense, DynamicIn, DynamicOut
export GaussianDense, GaussianDynamicIn, GaussianDynamicOut
export GaussianLogVarDense, GaussianLogVarDynamicIn, GaussianLogVarDynamicOut
export KumaraswamyDense, KumaraswamyDynamicIn, KumaraswamyDynamicOut
export BernoulliDense, BernoulliDynamicIn, BernoulliDynamicOut
export BernoulliLogitDense, BernoulliLogitDynamicIn, BernoulliLogitDynamicOut
export GumbelBernoulliLogitDense, GumbelBernoulliLogitDynamicIn, GumbelBernoulliLogitDynamicOut
export Chain

export initoptim!, update!, numparams
export AbstractTrainable, AbstractLayer, StaticLayer, StochasticLayer
export NeuralModel, train!, evaluate
