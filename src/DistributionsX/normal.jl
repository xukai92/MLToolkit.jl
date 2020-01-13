struct BroadcastedNormal{T<:Number,Tm<:AbstractArray{T},Ts<:AbstractArray{T}} <: ContinuousBatchDistribution
    m::Tm
    s::Ts
end

Broadcast.broadcastable(bd::BroadcastedNormal) = Ref(bd)

Normal(m::AbstractArray, s::AbstractArray) = BroadcastedNormal(m, s)

function rand(rng::AbstractRNG, bd::BroadcastedNormal, dims::Int...)
    return bd.m .+ bd.s .* randnsimilar(rng, bd.m, dims...)
end

function logpdf(bd::BroadcastedNormal{T}, x) where {T}
    diff = x .- bd.m
    v = bd.s.^2
    return -(log(2 * T(pi)) .+ log.(v) .+ diff .* diff ./ v) ./ 2
end

mean(bd::BroadcastedNormal) = bd.m

std(bd::BroadcastedNormal) = bd.s

var(bd::BroadcastedNormal) = bd.s.^2

mode(bd::BroadcastedNormal) = bd.m

function kldiv(bd1::BroadcastedNormal{T}, bd2::BroadcastedNormal{T}) where {T}
    diff = bd2.m .- bd1.m
    v1 = bd1.s.^2
    v2 = bd2.s.^2
    return (log.(v2) .- log.(v1) .- 1 .+ v1 ./ v2 .+ diff .* diff ./ v2) / 2
end

###



# struct BatchNormalLogVar{T}
#     μ::T    # mean
#     logΣ    # log-variance
# end

# function rand(dn::BatchNormalLogVar{T}) where {T}
#     ϵ = randnsimilar(dn.μ)
#     return dn.μ + exp.(dn.logΣ ./ 2) .* ϵ
# end

# function logpdf(dn::BatchNormalLogVar, x)
#     diff = x .- dn.μ
#     return -(log(2 * eltype(dn.μ)(pi)) .+ dn.logΣ .+ diff .* diff ./ exp.(dn.logΣ)) ./ 2
# end

# mean(bn::BatchNormalLogVar) = bn.μ
# mode(bn::BatchNormalLogVar) = bn.μ

# function kldiv(bnlv1::BatchNormalLogVar, bn2::BatchNormal)
#     if eltype(bnlv1.μ) != eltype(bn2.μ)
#         @warn "Float type are different for bnlv1 and bn2" eltype(bnlv1.μ) eltype(bn2.μ)
#     end
#     diff = bn2.μ .- bnlv1.μ
#     logΣ1 = bnlv1.logΣ
#     Σ2 = bn2.Σ
#     return (log.(Σ2) .- logΣ1 .- 1 .+ exp.(logΣ1) ./ Σ2 .+ diff .* diff ./ Σ2) ./ 2
# end

# using LinearAlgebra: det, tr, inv
# using Distributions: MvNormal

# """
#     kldiv(mvn1::MvNormal, mvn2::MvNormal)

# Compute ``KL(MvNormal_1||MvNormal_2)``.
# """
# function kldiv(mvn1::MvNormal, mvn2::MvNormal)
#     d = length(mvn1.μ)
#     diff = mvn2.μ .- mvn1.μ
#     Σ1 = Matrix(mvn1.Σ)
#     Σ2 = Matrix(mvn2.Σ)
#     return (log(det(Σ2)) - log(det(Σ1)) - d + tr(inv(Σ2) * Σ1) + diff' * inv(Σ2) * diff) ./ 2
# end
