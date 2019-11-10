### Neural sampler

struct NeuralSampler <: AbstractNeuralModel
    base
    f
    n_default::Int
end

NeuralSampler(base, f) = NeuralSampler(base, f, 1)

Flux.@functor NeuralSampler

Distributions.rand(rng::AbstractRNG, g::NeuralSampler, dims::Vararg{Int}=g.n_default) = g.f(rand(rng, g.base, dims...))

Distributions.rand(g::NeuralSampler, dims::Vararg{Int}=g.n_default) = rand(GLOBAL_RNG, g, dims...)

### Discriminator

struct Discriminator <: AbstractNeuralModel
    f
end

Flux.@functor Discriminator

(d::Discriminator)(x) = d.f(x)

### Projector

struct Projector <: AbstractNeuralModel
    f
    Dout::Int
end

Flux.@functor Projector

(p::Projector)(x) = p.f(x)
