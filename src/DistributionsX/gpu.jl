using CuArrays

# NOTE: The two functions below achive the following behaviour
# - Pass `rng` if it's of type `CuArrays.CURAND.RNG`;
# - Ignore `rng` and use the global of `CuArrays.CURAND` otherwise.
# The motivation is to avoid scalar operations on GPUs, which is the case when
# a CPU's RNG is used for inplace random number generation on GPUs.
function rsimilar(rng::CuArrays.CURAND.RNG, f!::Function, x::CuArray, n::Int)
    return rsimilar(rng, f!, x, n)
end
function rsimilar(_::AbstractRNG, f!::Function, x::CuArray, n::Int)
    return rsimilar(CuArrays.CURAND.generator(), f!, x, n)
end
