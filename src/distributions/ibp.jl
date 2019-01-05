using Distributions: Beta, Bernoulli

struct IBP
    α::Float64
    function IBP(α::Float64)
        @assert α > 0 "α is not positve"
        return new(α)
    end
end

"""
    rand(ibp::IBP, Kmax::Int64, n::Int64)

Sample from IBP using the stick-breaking construction.
"""
function rand(ibp::IBP, Kmax::Int64, n::Int64)
    ν = rand(Beta(ibp.α, 1), Kmax)
    p = break_stick_ibp(ν)
    Z = hcat(rand.(Bernoulli.(p), n)...)
    return Z
end

export IBP
