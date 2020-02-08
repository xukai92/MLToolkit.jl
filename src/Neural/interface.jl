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

function update!(opt, model::Trainable, data)
    ps = Flux.params(model)
    local res
    gs = gradient(ps) do
        res = loss(model, data)
        first(res)
    end
    Flux.Optimise.update!(opt, ps, gs)
    return res
end

loss(model::Trainable, v) = throw(MethodError(loss, model, v))
evaluate(model::Trainable, v) = throw(MethodError(evaluate, model, v))

getopt(m::Trainable) = m.opt
getiter(m::Trainable) = m.iter[]
setiter(m::Trainable, iter::Int) = (m.iter[] = iter)
increment!(m::Trainable) = (m.iter[] += 1)

function savemodel(model::Trainable, savepath::String; verbose=true)
    weights = Array.(Flux.params(cpu(model)))
    iter = getiter(model)
    BSON.bson(savepath, Dict(:iter => iter, :weights => weights))
    verbose && println("Saved model at iteration $iter to $savepath.")
    return savepath
end

function loadmodel!(model::Trainable, loadpath::String; verbose=true)
    modelfile = load(loadpath)
    Flux.loadparams!(model, modelfile[:weights])
    iter = modelfile[:iter]
    setiter!(model, iter)
    verbose && println("Loaded model at iteration $iter from $loadpath.")
end