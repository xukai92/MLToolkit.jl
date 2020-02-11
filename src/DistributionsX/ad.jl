using Tracker, Flux

### Zygote

Flux.Zygote.@nograd rsimilar

### Tracker

Base.eltype(x::Tracker.TrackedArray) = eltype(Tracker.data(x))
rsimilar(rng, f!, x::Tracker.TrackedArray, n) = rsimilar(rng, f!, Tracker.data(x), n)

function u2gumbelback(u, g)
    return 1 ./ exp.(-g) ./ (u .+ eps(u))
end

u2gumbel(u::Tracker.TrackedArray) = Tracker.track(u2gumbel, u)

Tracker.@grad function u2gumbel(u)
    g = u2gumbel(Tracker.data(u))
    return g, Δ -> (Δ .* u2gumbelback(u, g),)
end
