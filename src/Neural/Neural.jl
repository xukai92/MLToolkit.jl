module Neural

import ..MLToolkit: Reexport
Reexport.@reexport using Flux
using Random: AbstractRNG, GLOBAL_RNG

import Flux, Tracker, Distributions, ProgressMeter, BSON

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

function Flux.Zygote.pullback(f, p1::Flux.Params, p2::Flux.Params, prest::Flux.Params...)
    ps = (p1, p2, prest...)
    n = length(ps)
    cx = Flux.Zygote.Context()
    ys, back = Flux.Zygote._pullback(cx, f)
    @assert length(ys) == n
    caches = [copy(cx.cache) for i in 1:n]
    function back_i(Δ, i)
        cx.cache = caches[i]
        for pi in ps[i]
          Flux.Zygote.cache(cx)[pi] = nothing
        end
        back(Δ)
        Flux.Zygote.Grads(cx.cache)
    end
    ys, tuple((Δ -> back_i(Δ, i) for i in 1:n)...)
end

function Flux.Zygote.gradient(f, p1::Flux.Params, p2::Flux.Params, prest::Flux.Params...)
    ps = (p1, p2, prest...)
    ys, backs = Flux.Zygote.pullback(f, ps...)
    n = length(ys)
    makes(i) = tuple(map(j -> i == j ? Flux.Zygote.sensitivity(ys[i]) : zero(ys[i]), 1:n)...)
    return tuple([backs[i](makes(i)) for i in 1:n]...)
end

### 

abstract type AbstractNeuralModel end

abstract type Trainable <: AbstractNeuralModel end

function train!(
        model::Trainable, 
        dataloader, 
        n_epochs::Int=1;
        evalevery::Int=length(dataloader.train),
        cbeval::Function=function cbeval()
            @info "eval" evaluate(model, dataloader)... log_step_increment=0
        end,
        cbinit::Function=cbeval,
        saveevery::Int=evalevery,
        opt=getopt(model),
        savedir=getlogger(model).logdir,
        modelnamef::Function=iter -> "model.bson",
        refresh_zygote::Bool=true,
        verbose::Bool=true
)
    refresh_zygote && Flux.Zygote.refresh()
    progress = ProgressMeter.Progress(n_epochs * length(dataloader.train), 1, "Training")
    cbinit()
    for epoch in 1:n_epochs, batch in dataloader.train
        res = update!(opt, model, batch)
        verbose && @info "train" res...
        increment!(model)
        if evalevery > 0 && getiter(model) % evalevery == 0
            cbeval()
        end
        if saveevery > 0 && getiter(model) % saveevery == 0
            modelname = modelnamef(getiter(model))
            modelpath = "$savedir/$modelname.bson"
            savemodel(model, modelpath; verbose=verbose) 
        end
        ProgressMeter.next!(progress)
    end
end

# TODO: add backend option
function update!(opt, model::Trainable, batch)
    ps = Flux.params(model)
    local res
    gs = gradient(ps) do
        res = loss(model, batch)
        first(res)
    end
    Flux.Optimise.update!(opt, ps, gs)
    return res
end

loss(model::Trainable, v) = throw(MethodError(loss, model, v))
evaluate(model::Trainable, v) = throw(MethodError(evaluate, model, v))

getopt(model::Trainable) = model.opt

getlogger(model::Trainable) = hasproperty(model, :logger) ? model.logger : nothing

function getiter(model::Trainable)
    logger = getlogger(model)
    return isnothing(logger) ? model.iter[] : logger.global_step
end

function setiter!(model::Trainable, iter::Int)
    logger = getlogger(model)
    if isnothing(logger)
        model.iter[] = iter
    else
        logger.global_step = iter
    end
end

# Only manually increment if there is no logger (which automatically incremnted its own step already)
increment!(model::Trainable) = if isnothing(getlogger(model)) model.iter[] += 1 end

function savemodel(model::Trainable, modelpath="$(getlogger(model).logdir)/model.bson"; verbose=true)
    model_cpu = model |> cpu
    weights = Tracker.data.(Flux.params(model_cpu))
    BSON.bson(modelpath, Dict(:iter => getiter(model), :weights => weights))
    verbose && @info "Saved model at iteration $(getiter(model)) to $modelpath" log_step_increment=0
    return modelpath
end

function loadmodel!(model::Trainable, modelpath::String; verbose=true)
    file = BSON.load(modelpath)
    weights = file[:weights]
    Flux.loadparams!(model, weights)
    iter = file[:iter]
    setiter!(model, iter)
    verbose && @info "Loaded model at iteration $iter from $modelpath" log_step_increment=0
end

export Trainable, train!, update!, loss, evaluate, getopt, getlogger, getiter, setiter!, increment!, savemodel, loadmodel!

include("architecture.jl")
export DenseNet, ConvNet
include("gen.jl")
export NeuralSampler, Discriminator, Projector

end # module
