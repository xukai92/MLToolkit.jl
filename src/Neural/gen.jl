### Neural sampler

struct NeuralSampler <: AbstractNeuralModel
    base
    f
    n_default::Int
end

NeuralSampler(base, f) = NeuralSampler(base, f, 1)

Flux.@functor NeuralSampler

function Distributions.rand(rng::AbstractRNG, g::NeuralSampler, n::Int=g.n_default)
    z = rand(g.base, n)
    return g.f(usegpu[] ? gpu(z) : z)
end

Distributions.rand(g::NeuralSampler, n::Int) = rand(GLOBAL_RNG, g, n)

### Discriminator

struct Discriminator <: AbstractNeuralModel
    f
end

Flux.@functor Discriminator

(d::Discriminator)(x) = d.f(x)

### Projector

struct Projector <: AbstractNeuralModel
    Dfx::Int
    f
end

Flux.@functor Projector

(p::Projector)(x) = p.f(x)
