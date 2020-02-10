using MLDataUtils: eachbatch
using DrWatson: DrWatson, tagsave, load
using Flux: Flux, gpu, cpu, params, loadparams!, Optimise
using ProgressMeter: Progress, next!

abstract type Trainable <: AbstractNeuralModel end

loss(m::Trainable, data) = throw(MethodError(m, data))

function update!(opt, m::Trainable, data)
    ps = params(m)
    local info
    gs = gradient(ps) do
        info = loss(m, data)
        info.loss
    end
    Optimise.update!(opt, ps, gs)
    return info
end

# TODO: How to handle `train!` without `opt`?
train!(opt, m::Trainable, dataset, n_epochs::Int, batch_size::Int; kwargs...) = 
    train!(opt, m, eachbatch(dataset; size=batch_size), n_epochs; kwargs...)
function train!(
    opt, m::T, dataiter, n_epochs::Int;
    verbose::Bool=true, is_refresh::Bool=false,
    prepare::Function=(data -> gpu(data)),
    evalevery::Int=length(dataiter), cbeval::Union{Nothing, Function}=nothing,
    saveevery::Int=length(dataiter), savedir::Union{Nothing, String}=nothing,
) where {T<:Trainable}
    is_refresh && Flux.Zygote.refresh()
    progress = Progress(n_epochs * length(dataiter); desc="Training: ")
    for epoch in 1:n_epochs, data in dataiter
        # NOTE: It's very hard to unify `cbeval` within `update!`
        #       not only because the signature could be `cbeval(data)`
        #       but also that we do not gurantee to get internal variables out of `update!`.
        info = update!(opt, m, prepare(data))
        let l = info.loss
            (isnan(l) || isinf(l)) && error("Loss has numeric error; loss=$l.")
        end
        next!(progress)
        # Logging
        if :step in fieldnames(T)
            step = (m.step[] += 1)
        else 
            step = progress.counter 
        end
        if evalevery > 0 && (step % evalevery == 0 || step % length(dataiter) == 0 ) && !isnothing(cbeval)
            verbose && @info "eval" step=step cbeval()... commit=false
        end
        if saveevery > 0 && (step % saveevery == 0 || step % length(dataiter) == 0 ) && !isnothing(savedir)
            saveparams(m, joinpath(savedir, "m-$step.bson"); verbose=verbose)
        end
        verbose && @info "train" step=step info...
    end
end

function saveparams(m::Trainable, savepath::String; verbose=true)
    weights = Array.(Flux.params(cpu(m)))
    tagsave(savepath, Dict(:weights => weights, :step => m.step[]); safe=true)
    verbose && println("Saved m at step $step to $savepath.")
end

function Flux.loadparams!(m::T, loadpath::String; verbose=true) where {T<:Trainable}
    modelfile = load(loadpath)
    Flux.loadparams!(m, modelfile[:weights])
    step = modelfile[:step]
    if :step in fieldnames(T)
        m.step[] = step
    else 
        @warn "`$T` doesn't have field `step`."
    end
    verbose && println("Loaded m at step $step from $loadpath.")
end
