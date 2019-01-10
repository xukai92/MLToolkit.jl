using Distributions: Beta, Bernoulli, Poisson

struct IBP{T<:Real}
    α::T
    function IBP{T}(α) where {T<:Real}
        @assert α > 0 "α is not positve"
        return new{T}(α)
    end
end
IBP(α::T) where {T<:Real} = IBP{T}(α)

"""
    rand(ibp::IBP, Kmax::Int, n::Int)

Sample from IBP using the stick-breaking construction.
The returned sample is of size `n`-by-`Kmax`,
which is not a Julia convension by a math convension.
"""
function rand(ibp::IBP, Kmax::Int, n::Int)
    ν = rand(Beta(ibp.α, 1), Kmax)
    p = break_stick_ibp(ν)
    Z = hcat(rand.(Bernoulli.(p), n)...)
    return Z
end

"""
    rand(ibp::IBP, n::Int)

Sample from IBP using the buffet metaphor.
The returned sample is of size `n`-by-`Kmax`,
which is not a Julia convension by a math convension.
"""
function rand(ibp::IBP, n::Int)
    k_init = rand(Poisson(ibp.α))
    Z = zeros(Integer, n, k_init)
    Z[1,:] .= 1
    for i = 2:n
        for k = 1:size(Z, 2)
            mk = sum(Z[:,k])
            Z[i,k] = rand(Bernoulli(mk / i))
        end
        k_new = rand(Poisson(ibp.α / i))
        if k_new > 0
            Z = hcat(Z, zeros(Integer, n, k_new))
            Z[i,end-k_new+1:end] .= 1
        end
    end
    return Z
end
