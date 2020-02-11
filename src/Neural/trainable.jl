using MLDataUtils: eachbatch
using DrWatson: DrWatson, tagsave, load
using Flux: Flux, gpu, cpu, params, loadparams!, Optimise
using ProgressMeter: ProgressMeter, Progress, next!

# Interface and extensible functions
abstract type Trainable <: AbstractNeuralModel end
prepare(m::Trainable, data) = data
prepare(m::Trainable, data::AbstractArray) = gpu(data)

function update!(opt, m::Trainable, data)
    ps = params(m)
    local info, loss
    gs = gradient(ps) do
        info = m(data)
        loss = first(info)
        loss
    end
    (isnan(loss) || isinf(loss)) && error("Loss has numeric error; loss=$loss.")
    Optimise.update!(opt, ps, gs)
    return info
end

function ProgressMeter.next!(m::T) where {T<:Trainable}
    if :step in fieldnames(T)
        m.step[] += 1
    end
end

# TODO: How to handle `train!` without `opt`?
train!(opt, m::Trainable, trainset, n_epochs::Int, batch_size::Int; kwargs...) = 
    train!(opt, m, eachbatch(trainset; size=batch_size), n_epochs; kwargs...)
function train!(
    opt, m::T, trainiter, n_epochs::Int;
    verbose::Bool=true, is_refresh::Bool=false,
    evalevery::Int=length(trainiter), cbeval::Union{Nothing, Function}=nothing,
    saveevery::Int=length(trainiter), savedir::Union{Nothing, String}=nothing,
) where {T<:Trainable}
    is_refresh && Flux.Zygote.refresh()
    progress = Progress(n_epochs * length(trainiter); desc="Training: ")
    for epoch in 1:n_epochs, data in trainiter
        # NOTE: It's very hard to unify `cbeval` within `update!`
        #       not only because the signature could be `cbeval(data)`
        #       but also that we do not gurantee to get internal variables out of `update!`.
        info = update!(opt, m, prepare(m, data))
        # Logging
        step = :step in fieldnames(T) ? m.step[] : progress.counter 
        if evalevery > 0 && (step % evalevery == 0 || step % length(trainiter) == 0 ) && !isnothing(cbeval)
            verbose && @info "eval" step=step cbeval()... commit=false
        end
        if saveevery > 0 && (step % saveevery == 0 || step % length(trainiter) == 0 ) && !isnothing(savedir)
            modelname = "model-$step.bson"
            saveparams(m, joinpath(savedir, modelname); verbose=verbose)
            if epoch == n_epochs && step % length(trainiter) == 0
                symlink(joinpath(savedir, modelname), joinpath(savedir, "model.bson"))
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
