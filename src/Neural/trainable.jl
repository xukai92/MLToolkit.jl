using MLDataUtils: eachbatch
using DrWatson: DrWatson, tagsave, load
using Flux: gpu, cpu, params, loadparams!, Optimise

abstract type Trainable <: AbstractNeuralModel end

loss(model::Trainable, data) = throw(MethodError(loss, model, data))
eval(model::Trainable, dataiter) = throw(MethodError(eval, model, dataiter))

function step!(opt, model::Trainable, data)
    ps = params(model)
    local res
    gs = gradient(ps) do
        res = loss(model, data)
        first(res)
    end
    Optimise.update!(opt, ps, gs); model.step[] += 1
    return res
end

train!(opt, model::Trainable, dataset, n_epochs::Int, batch_size::Int; kwargs...) = 
    train!(opt, model, eachbatch(dataset; size=batch_size), n_epochs; kwargs...)

function train!(
    opt,
    model::Trainable,
    dataiter,
    n_epochs::Int;
    refresh_zygote::Bool=true,
    prepare::Function=(data -> gpu(data)),
    evalevery::Int=0,
    saveevery::Int=length(dataiter),
    savedir::Union{Nothing, String}=nothing,
    verbose::Bool=true,
)
    refresh_zygote && Flux.Zygote.refresh()
    progress = ProgressMeter.Progress(n_epochs * length(dataiter), 1, "Training")
    for epoch in 1:n_epochs
        is_numeric_error = false
        for (iter, data) in enumerate(dataiter)
            data = prepare(data)
            res = step!(opt, model, data)
            if isnan(first(res)) || isinf(first(res))
                is_numeric_error = true
                break
            end
            if evalevery > 0 && model.step[] % evalevery == 0
                res = eval(model, dataiter) # or eval(model, data)
                verbose && @info "eval" step=model.step[] res... commit=false
            end
            if saveevery > 0 && model.step[] % saveevery == 0 && !isnothing(savedir)
                save(model, joinpath(savedir, "model-$(model.step[]).bson"))
            end
            verbose && @info "train" step=model.step[] res...
            ProgressMeter.next!(progress)
        end
        is_numeric_error && break
    end
end

function saveparams(model::Trainable, savepath::String; verbose=true)
    weights = Array.(Flux.params(cpu(model)))
    tagsave(savepath, Dict(:weights => weights, :step => model.step[]); safe=true)
    verbose && println("Saved model at step $step to $savepath.")
    return savepath
end

function loadparams!(model::Trainable, loadpath::String; verbose=true)
    modelfile = load(loadpath)
    Flux.loadparams!(model, modelfile[:weights])
    model.step[] = modelfile[:step]
    verbose && println("Loaded model at step $(model.step[]) from $loadpath.")
end
