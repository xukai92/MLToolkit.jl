struct IBP{T<:Real}
    α::T
    # TODO: IBP{T} -> IBP below and Line 11 could be removed
    function IBP{T}(α) where {T<:Real}
        @assert α > 0 "α is not positve"
        return new{T}(α)
    end
end
IBP(α::T) where {T<:Real} = IBP{T}(α)

"""
    rand(ibp::IBP, n::Int, k_max::Int)

Sample from IBP using the stick-breaking construction.
The returned sample is of size `n`-by-`k_max`,
which is not a Julia convension by a math convension.
"""
function rand(ibp::IBP, n::Int, k_max::Int)
    ν = rand(Distributions.Beta(ibp.α, 1), k_max)
    p = break_stick_ibp(ν)
    Z = hcat(rand.(Distributions.Bernoulli.(p), n)...)
    return Z
end

"""
    rand(ibp::IBP, n::Int)

Sample from IBP using the buffet metaphor.
The returned sample is of size `n`-by-`Kmax`,
which is not a Julia convension by a math convension.
"""
function rand(ibp::IBP, n::Int)
    k_init = rand(Distributions.Poisson(ibp.α))
    Z = zeros(Int, n, k_init)
    Z[1,:] .= 1
    for i = 2:n
        for k = 1:size(Z, 2)
            mk = sum(Z[:,k])
            Z[i,k] = rand(Distributions.Bernoulli(mk / i))
        end
        k_new = rand(Distributions.Poisson(ibp.α / i))
        if k_new > 0
            Z = hcat(Z, zeros(Int, n, k_new))
            Z[i,end-k_new+1:end] .= 1
        end
    end
    return Z
end
