module Neural

import Flux

### Tracker support

params(m) = m |> Flux.params |> Tracker.Params

# https://github.com/FluxML/Flux.jl/blob/bdeb9c6d584668c7cef1ce71caf659d611c86d65/src/optimise/train.jl#L9-L18
function apply!(opt, xs::Tracker.Params, Δs)
    for x in xs
        Δs[x] == nothing && continue
        x.data .-= Flux.Optimise.apply!(opt, Tracker.data(x), Tracker.data(Δs[x]))
    end
end

using Tracker: Params, losscheck, @interrupts, Grads, tracker, extract_grad!, zero_grad!, grad

function Tracker.gradient(f, xs::Params; once=true)
    l = f()
    losscheck(l)
    @interrupts back!(l; once=once)
    gs = Grads()
    for x in xs
        gs[tracker(x)] = extract_grad!(x)
    end
    return gs
end

track(m) = fmap(x -> x isa AbstractArray ? Tracker.TrackedArray(x) : x, m)

export track

###

nparams(m) = sum(prod.(size.(Flux.params(m))))

export nparams

end # module
