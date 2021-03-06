abstract type AbstractBroadcastedNormal{T} <: ContinuousBatchDistribution end

function rand(rng::AbstractRNG, bd::AbstractBroadcastedNormal, dims::Int...)
    return bd.m .+ std(bd) .* randnsimilar(rng, bd.m, dims...)
end

function logpdf(bd::AbstractBroadcastedNormal{T}, x) where {T}
    diff = x .- bd.m
    return -(log(2 * T(pi)) .+ logvar(bd) .+ diff .* diff ./ var(bd)) / 2
end

mean(bd::AbstractBroadcastedNormal) = bd.m
mode(bd::AbstractBroadcastedNormal) = bd.m

_vlogv(bd::AbstractBroadcastedNormal) = (var(bd), logvar(bd))

function kldiv(bd1::AbstractBroadcastedNormal{T}, bd2::AbstractBroadcastedNormal{T}) where {T}
    diff = bd2.m .- bd1.m
    v1, logv1 = _vlogv(bd1)
    v2, logv2 = _vlogv(bd2)
    return (logv2 .- logv1 .- 1 .+ v1 ./ v2 .+ diff .* diff ./ v2) / 2
end

"""
Broadcasted Normal distribution with standard deviation.
"""
struct BroadcastedNormalStd{T<:Number,Tm<:AbstractArray{T},Ts<:AbstractArray{T}} <: AbstractBroadcastedNormal{T}
    m::Tm
    s::Ts
end

Broadcast.broadcastable(bd::BroadcastedNormalStd) = Ref(bd)

Normal(m::AbstractArray, s::AbstractArray) = BroadcastedNormalStd(m, s)
NormalStd(m::AbstractArray, s::AbstractArray) = BroadcastedNormalStd(m, s)

   std(bd::BroadcastedNormalStd) = bd.s
   var(bd::BroadcastedNormalStd) = bd.s.^2
logvar(bd::BroadcastedNormalStd) = 2log.(bd.s)

"""
Broadcasted Normal distribution with variance.
"""
struct BroadcastedNormalVar{T<:Number,Tm<:AbstractArray{T},Tv<:AbstractArray{T}} <: AbstractBroadcastedNormal{T}
    m::Tm
    v::Tv
end

Broadcast.broadcastable(bd::BroadcastedNormalVar) = Ref(bd)

NormalVar(m::AbstractArray, v::AbstractArray) = BroadcastedNormalVar(m, v)

   std(bd::BroadcastedNormalVar) = sqrt.(bd.v)
   var(bd::BroadcastedNormalVar) = bd.v
logvar(bd::BroadcastedNormalVar) = log.(bd.v)

"""
Broadcasted Normal distribution with log standard deviation.
"""
struct BroadcastedNormalLogStd{T<:Number,Tm<:AbstractArray{T},Ts<:AbstractArray{T}} <: AbstractBroadcastedNormal{T}
       m::Tm    # mean
    logs::Ts    # log std
end

Broadcast.broadcastable(bd::BroadcastedNormalLogStd) = Ref(bd)

NormalLogStd(m::AbstractArray, logs::AbstractArray) = BroadcastedNormalLogStd(m, logs)

   std(bd::BroadcastedNormalLogStd) = exp.(bd.logs)
   var(bd::BroadcastedNormalLogStd) = exp.(2bd.logs)
logvar(bd::BroadcastedNormalLogStd) = 2bd.logs

function _vlogv(bd::BroadcastedNormalLogStd)
    logv = logvar(bd)
    v = exp.(logv)
    return v, logv
end

"""
Broadcasted Normal distribution with log variance.
"""
struct BroadcastedNormalLogVar{T<:Number,Tm<:AbstractArray{T},Tv<:AbstractArray{T}} <: AbstractBroadcastedNormal{T}
       m::Tm    # mean
    logv::Tv    # log std
end

Broadcast.broadcastable(bd::BroadcastedNormalLogVar) = Ref(bd)

NormalLogVar(m::AbstractArray, logv::AbstractArray) = BroadcastedNormalLogVar(m, logv)

   std(bd::BroadcastedNormalLogVar) = exp.(bd.logv ./ 2)
   var(bd::BroadcastedNormalLogVar) = exp.(bd.logv)
logvar(bd::BroadcastedNormalLogVar) = bd.logv

function _vlogv(bd::BroadcastedNormalLogVar)
    logv = logvar(bd)
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
