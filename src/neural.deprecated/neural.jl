import Knet: update!, train!

abstract type AbstractTrainable end
abstract type AbstractLayer <: AbstractTrainable end
# TODO: rename StaticLayer -> AbstractStaticLayer and StochasticLayer -> AbstractStochasticLayer
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
Dynamic Adam - an Adam that can initialise first and second momentum during optimization.
This is used to deal dynamic models.
"""
mutable struct DynamicAdam
    adam::Knet.Adam
    isreset::Bool
end

DynamicAdam(isreset=true; kwargs...) = DynamicAdam(Knet.Adam(; kwargs...), isreset)

for T in (Array{Float32},Array{Float64},Knet.KnetArray{Float32},Knet.KnetArray{Float64}); @eval begin
    function update!(w::$T, g::$T, p::DynamicAdam)
        if !(p.adam.fstm === nothing) && length(g) > length(p.adam.fstm)
            F = eltype($T)
            if p.isreset
                p.adam.fstm = $T(zeros(F, size(g)...))
                p.adam.scndm = $T(zeros(F, size(g)...))
            else
                # Vector
                if length(size(g)) == 1
                    d_pad = length(g) - length(p.adam.fstm)
                    p.adam.fstm = vcat([p.adam.fstm, $T(zeros(F, d_pad))]...)
                    p.adam.scndm = vcat([p.adam.scndm, $T(zeros(F, d_pad))]...)
                # Matrix
                else
                    # Expand rows
                    if size(g, 1) > size(p.adam.fstm, 1)
                        d_pad = size(g, 1) - size(p.adam.fstm, 1)
                        p.adam.fstm = vcat([p.adam.fstm, $T(zeros(F, d_pad, size(g, 2)))]...)
                        p.adam.scndm = vcat([p.adam.scndm, $T(zeros(F, d_pad, size(g, 2)))]...)
                    # Expand cols
                    else
                        d_pad = size(g, 2) - size(p.adam.fstm, 2)
                        p.adam.fstm = hcat([p.adam.fstm, $T(zeros(F, size(g, 1), d_pad))]...)
                        p.adam.scndm = hcat([p.adam.scndm, $T(zeros(F, size(g, 1), d_pad))]...)
                    end
                end
            end
        end
        update!(w, g, p.adam)
    end
end; end

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
function train!(model::NeuralModel, dataloader; kwargs...)
    loss_list = []
    # Pop `:epoch` out from from `kwargs` if exist
    local epoch = nothing
    if :epoch in keys(kwargs)
        epoch = kwargs[:epoch]
        kwargs = Dict(kwargs...)
        # NOTE: the line below is uncommented because `:epoch` is useful for annealing
        #       so we need to deal with possible unwanted keyword agurments manually
        # pop!(kwargs, :epoch)
    end
    for (i, data_batch) in enumerate(dataloader)
        # Calculate the iteration till now if given the epoch
        if epoch != nothing
            kwargs[:iter] = length(dataloader) * (epoch - 1) + i
        end
        losstape = Knet.@diff model(data_batch, Val(:true); kwargs...)
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

export initoptim!, update!, DynamicAdam, numparams
export AbstractTrainable, AbstractLayer, StaticLayer, StochasticLayer
export NeuralModel, train!, evaluate

include("activations.jl")
export log1pexp, logexpm1, softplus, invsoftplus, leaky_relu
include("layers.jl")
export Dense, LazyDense, DynamicIn, DynamicOut, Chain
include("stochastic_layers.jl")
# TODO: somehow make below automatic
export GaussianDense, GaussianLazyDense, GaussianDynamicIn, GaussianDynamicOut
export GaussianLogVarDense, GaussianLogVarLazyDense, GaussianLogVarDynamicIn, GaussianLogVarDynamicOut
export KumaraswamyDense, KumaraswamyLazyDense, KumaraswamyDynamicIn, KumaraswamyDynamicOut
export BernoulliDense, BernoulliLazyDense, BernoulliDynamicIn, BernoulliDynamicOut
export BernoulliLogitDense, BernoulliLogitLazyDense, BernoulliLogitDynamicIn, BernoulliLogitDynamicOut
export GumbelBernoulliLogitDense, GumbelBernoulliLogitLazyDense, GumbelBernoulliLogitDynamicIn, GumbelBernoulliLogitDynamicOut
include("sbc.jl")
export MeanFieldSBC, StructuredSBC

include("rho.jl")
export Rho, updategrad!
