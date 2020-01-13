abstract type AbstractNormal{T} <: ContinuousBatchDistribution end

function rand(rng::AbstractRNG, bd::AbstractNormal, dims::Int...)
    return bd.m .+ std(bd) .* randnsimilar(rng, bd.m, dims...)
end

function logpdf(bd::AbstractNormal{T}, x) where {T}
    diff = x .- bd.m
    return -(log(2 * T(pi)) .+ logvar(bd) .+ diff .* diff ./ var(bd)) ./ 2
end

mean(bd::AbstractNormal) = bd.m
mode(bd::AbstractNormal) = bd.m

function kldiv(bd1::AbstractNormal{T}, bd2::AbstractNormal{T}) where {T}
    diff = bd2.m .- bd1.m
    v1, logv1 = _vlogv(bd1)
    v2, logv2 = _vlogv(bd2)
    return (logv2 .- logv1 .- 1 .+ v1 ./ v2 .+ diff .* diff ./ v2) / 2
end

"""
Broadcasted Normal distribution.
"""
struct BroadcastedNormal{T<:Number,Tm<:AbstractArray{T},Ts<:AbstractArray{T}} <: AbstractNormal{T}
    m::Tm
    s::Ts
end

Broadcast.broadcastable(bd::BroadcastedNormal) = Ref(bd)

Normal(m::AbstractArray, s::AbstractArray) = BroadcastedNormal(m, s)

   std(bd::BroadcastedNormal) = bd.s
   var(bd::BroadcastedNormal) = bd.s.^2
logvar(bd::BroadcastedNormal) = 2log.(bd.s)

_vlogv(bd::BroadcastedNormal) = (var(bd), logvar(bd))

"""
Broadcasted Normal distribution with log standard deviation.
"""
struct BroadcastedNormalLogStd{T<:Number,Tm<:AbstractArray{T},Ts<:AbstractArray{T}} <: AbstractNormal{T}
       m::Tm    # mean
    logs::Ts    # log std
end

Broadcast.broadcastable(bd::BroadcastedNormalLogStd) = Ref(bd)

NormalLogStd(m::AbstractArray, logs::AbstractArray) = BroadcastedNormalLogStd(m, logs)

   std(bd::BroadcastedNormalLogStd) = exp.(bd.logs)
   var(bd::BroadcastedNormalLogStd) = exp.(2bd.logs)
logvar(bd::BroadcastedNormalLogStd) = 2bd.logs

function _vlogv(bd::BroadcastedNormalLogStd)
    logv = 2bd.logs
    v = exp.(logv)
    return v, logv
end

###

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
