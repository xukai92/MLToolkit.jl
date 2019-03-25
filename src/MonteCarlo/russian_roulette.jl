"""
Approximate the summation below using single sample Russian roulette sampling.

S = \\sum_{i=1}^{\\infty} X_i
"""
function roll(X::Function, p::DiscreteUnivariateDistribution, n_mc::Int=1)
    taus = filter(t -> t > 0, rand(p, n_mc))
    if length(taus) == 0
        return 0
    end
    tau_min, tau_max = extrema(taus)
    ws = 1 ./ ccdf.(Ref(p), 0:tau_max-1)
    if tau_max > tau_min
        rws = reverse(cumsum(reverse(counts(taus)))) / n_mc
        ws[tau_min:tau_max] .*= rws
    end
    return sum(X.(1:tau_max) .* ws)
end


"""
Approximate the summation below using single sample Russian roulette sampling.

S = \\sum_{i=1}^{\\infty} m_i T_i
"""
function roll_expectation(T::Function, m::DiscreteUnivariateDistribution, p::DiscreteUnivariateDistribution, n_mc::Int=1)
    if m == p
        return roll_expectation(T, m)
    end
    taus = filter(t -> t > 0, rand(p, n_mc))
    if length(taus) == 0
        return 0
    end
    tau_min, tau_max = extrema(taus)
    ws = pdf.(Ref(m), 1:tau_max) ./ ccdf.(Ref(p), 0:tau_max-1)
    if tau_max > tau_min
        rws = reverse(cumsum(reverse(counts(taus)))) / n_mc
        ws[tau_min:tau_max] .*= rws
    end
    return sum(T.(1:tau_max) .* ws)
end

function roll_expectation(T::Function, p::LogitNPD, n_mc::Int=1)
    taus = rand(p, n_mc)
    tau_min, tau_max = extrema(taus)
    ws = 1 .- getρ.(Ref(p), 1:tau_max)
    if tau_max > tau_min
        rws = reverse(cumsum(reverse(counts(taus)))) / n_mc
        ws[tau_min:tau_max] .*= rws
    end
    return sum(T.(1:tau_max) .* ws)
end
