for T in [
    GumbelSoftmax,
    GumbelBernoulli,
    GumbelBernoulliLogit,
    BatchBernoulli,
    BatchBernoulliLogit,
]
    @eval Flux.@functor $T
end

Flux.gpu_adaptor(x::UniformNoise) = x
Flux.adapt(::Type{Array}, x::UniformNoise) = x
