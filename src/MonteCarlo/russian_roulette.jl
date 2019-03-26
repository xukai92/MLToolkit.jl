# TODO: somehow merge the common codes below

"""
Approximate the summation below using single sample Russian roulette sampling.

S = \\sum_{i=1}^{\\infty} X_i
"""
roll(X::Function, p::DiscreteUniDist, n_mc::Int=1) = roll(X, p, rand(p, n_mc))
function roll(X::Function, p::DiscreteUniDist, τs::AbstractVector{Int})
    n_mc = length(τs)
    τs = filter(t -> t > 0, τs)
    if length(τs) == 0
        return 0
    end
    τ_min, τ_max = extrema(τs)
    ws = 1 ./ ccdf.(Ref(p), 0:τ_max-1)
    if τ_max > τ_min
        rws = reverse(cumsum(reverse(counts(τs)))) / n_mc
        ws[τ_min:τ_max] .*= rws
    end
    return sum(X.(1:τ_max) .* ws)
end

"""
Approximate the summation below using single sample Russian roulette sampling.

S = \\sum_{i=1}^{\\infty} m_i T_i
"""
roll_expectation(T::Function, m::DiscreteUniDist, p::DiscreteUniDist, n_mc::Int=1) = roll_expectation(T, m, p, rand(p, n_mc))
function roll_expectation(T::Function, m::DiscreteUniDist, p::DiscreteUniDist, τs::AbstractVector{Int})
    n_mc = length(τs)
    τs = filter(t -> t > 0, τs)
    if length(τs) == 0
        return 0
    end
    τ_min, τ_max = extrema(τs)
    ws = pdf.(Ref(m), 1:τ_max) ./ ccdf.(Ref(p), 0:τ_max-1)
    if τ_max > τ_min
        rws = reverse(cumsum(reverse(counts(τs)))) / n_mc
        ws[τ_min:τ_max] .*= rws
    end
    return sum(T.(1:τ_max) .* ws)
end

roll_expectation(T::Function, p::LogitNPD, n_mc::Int=1) = roll_expectation(T, p, rand(p, n_mc))
function roll_expectation(T::Function, p::LogitNPD, τs::AbstractVector{Int})
    n_mc = length(τs)
    τ_min, τ_max = extrema(τs)
    ws = 1 .- getρ.(Ref(p), 1:τ_max)
    if τ_max > τ_min
        rws = reverse(cumsum(reverse(counts(τs)))) / n_mc
        ws[τ_min:τ_max] .*= rws
    end
    return sum(T.(1:τ_max) .* ws)
end
