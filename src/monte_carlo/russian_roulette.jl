"""
Approximate the summation below using single sample Russian roulette sampling.

S = \\sum_{i=1}^{\\infty} X_i
"""
function roll(X::Function, p::Distributions.DiscreteUnivariateDistribution)
    tau = rand(p)
    if tau == 0
        return 0
    else
        return sum(X.(1:tau) ./ ccdf.(Ref(p), 0:tau-1))
    end
end

"""
Approximate the summation below using single sample Russian roulette sampling.

S = \\sum_{i=1}^{\\infty} m_i T_i
"""
function roll_exp(T::Function, m::Distributions.DiscreteUnivariateDistribution, p::Distributions.DiscreteUnivariateDistribution)
    if m == p
        return roll_exp(T, m)
    end
    tau = rand(p)
    if tau == 0
        return 0
    end
    return sum(T.(1:tau) .* pdf.(Ref(m), 1:tau) ./ ccdf.(Ref(p), 0:tau-1))
end

function roll_exp(T::Function, p::LogitNPD)
    tau = rand(p)
    return sum(T.(1:tau) .* (1 .- getρ.(Ref(p), 1:tau)))
end
