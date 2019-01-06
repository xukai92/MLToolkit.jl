using Distributions: Beta, Bernoulli

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

Sample from IBP using the stick-breaking construction. The returned sample is of size `n`-by-`Kmax`, which is not a Julia convension by a math convension.
"""
function rand(ibp::IBP, Kmax::Int, n::Int)
    ν = rand(Beta(ibp.α, 1), Kmax)
    p = break_stick_ibp(ν)
    Z = hcat(rand.(Bernoulli.(p), n)...)
    return Z
end

export IBP
