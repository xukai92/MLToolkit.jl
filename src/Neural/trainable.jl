using MLDataUtils: eachbatch
using DrWatson: DrWatson, tagsave, load
using Flux: Flux, gpu, cpu, params, loadparams!, Optimise
using ProgressMeter: ProgressMeter, Progress, next!

# Interface and extensible functions
abstract type Trainable <: AbstractNeuralModel end
loss(m::Trainable, data) = throw(MethodError(m, data))
prepare(m::Trainable, data) = data
prepare(m::Trainable, data::AbstractArray) = gpu(data)

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

function ProgressMeter.next!(m::T) where {T<:Trainable}
    if :step in fieldnames(T)
        m.step[] += 1
    end
end

# TODO: How to handle `train!` without `opt`?
train!(opt, m::Trainable, dataset, n_epochs::Int, batch_size::Int; kwargs...) = 
    train!(opt, m, eachbatch(dataset; size=batch_size), n_epochs; kwargs...)
function train!(
    opt, m::T, dataiter, n_epochs::Int;
    verbose::Bool=true, is_refresh::Bool=false,
    evalevery::Int=length(dataiter), cbeval::Union{Nothing, Function}=nothing,
    saveevery::Int=length(dataiter), savedir::Union{Nothing, String}=nothing,
) where {T<:Trainable}
    is_refresh && Flux.Zygote.refresh()
    progress = Progress(n_epochs * length(dataiter); desc="Training: ")
    for epoch in 1:n_epochs, data in dataiter
        # NOTE: It's very hard to unify `cbeval` within `update!`
        #       not only because the signature could be `cbeval(data)`
        #       but also that we do not gurantee to get internal variables out of `update!`.
        info = update!(opt, m, prepare(m, data))
        let l = info.loss
            (isnan(l) || isinf(l)) && error("Loss has numeric error; loss=$l.")
        end
        # Logging
        step = :step in fieldnames(T) ? m.step[] : progress.counter 
        if evalevery > 0 && (step % evalevery == 0 || step % length(dataiter) == 0 ) && !isnothing(cbeval)
            verbose && @info "eval" step=step cbeval()... commit=false
        end
        if saveevery > 0 && (step % saveevery == 0 || step % length(dataiter) == 0 ) && !isnothing(savedir)
            modelname = "model-$step.bson"
            saveparams(m, joinpath(savedir, modelname); verbose=verbose)
            if epoch == n_epochs && step % length(dataiter) == 0
                symlink(joinpath(savedir, "model.bson"), joinpath(savedir, modelname))
            end
        end
        verbose && @info "train" step=step info...
        # Progress
        next!.((progress, m)) 
    end
end

function saveparams(m::T, savepath::String; verbose=true) where {T<:Trainable}
    savedict = Dict(:weights => Array.(params(m)))
    if :step in fieldnames(T)
        savedict[:step] = m.step[]
    end
    tagsave(savepath, savedict; safe=true)
    verbose && println("Saved $T to $savepath.")
end

function Flux.loadparams!(m::T, loadpath::String; verbose=true) where {T<:Trainable}
    modelfile = load(loadpath)
    loadparams!(m, modelfile[:weights])
    if :step in fieldnames(T)
        m.step[] = modelfile[:step]
    end
    verbose && println("Loaded $T from $loadpath.")
end
