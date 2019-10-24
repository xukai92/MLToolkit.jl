# Neural stick-breaking processe

struct MeanFieldSBC <: StochasticLayer
    kuma
    gumbel
end

function MeanFieldSBC(i_dim::Int, k_init::Int=0; kwargs...)
    kuma = KumaraswamyLazyDense(i_dim, k_init; kwargs...)
    gumbel = GumbelBernoulliLogitLazyDense(i_dim, k_init; kwargs...)
    return MeanFieldSBC(kuma, gumbel)
end

function (enc::MeanFieldSBC)(i, k::Int; lowerbound=(FT == Float64 ? FT(0.1) : FT(0.2)))
    dist_kuma = enc.kuma(i, k)
    dist_nu = BatchKumaraswamy(dist_kuma.a .+ lowerbound, dist_kuma.b .+ lowerbound)
    dist_gumbel = enc.gumbel(i, k)
    return dist_nu, dist_gumbel
end

struct StructuredSBC <: StochasticLayer
    a
    b
    gumbel
    α
    β
end

"""
Helper function to initialise StructuredSBC with an IBP.
"""
function StructuredSBC(ibp::IBP{T}, i_dim::Int, k_init::Int=0; kwargs...) where {T<:AbstractFloat}
    return StructuredSBC(i_dim, k_init, ibp.α, one(T); kwargs...)
end

function StructuredSBC(i_dim::Int, k_init::Int=0, α::AbstractFloat=one(FT), β::AbstractFloat=one(FT); kwargs...)
    a = Knet.Param(AT(zeros(FT, k_init, 1) .+ invsoftplus(α)))
    b = Knet.Param(AT(zeros(FT, k_init, 1) .+ invsoftplus(β)))
    gumbel = GumbelBernoulliLogitLazyDense(i_dim, k_init; kwargs...)
    return StructuredSBC(a, b, gumbel, α, β)
end

function (enc::StructuredSBC)(i, k::Int=size(enc.a, 1); α::AbstractFloat=enc.α, β::AbstractFloat=enc.β, lowerbound=(FT == Float64 ? FT(0.1) : FT(0.2)))
    if k > size(enc.a, 1)
        k_diff = k - size(enc.a, 1)
        enc.a.value = vcat([enc.a.value, AT(zeros(FT, k_diff, 1) .+ invsoftplus(α))]...)
        enc.b.value = vcat([enc.b.value, AT(zeros(FT, k_diff, 1) .+ invsoftplus(β))]...)
    end
    dist_nu = BatchKumaraswamy(softplus.(enc.a[1:k,:]) .+ lowerbound,
                               softplus.(enc.b[1:k,:]) .+ lowerbound)
    dist_gumbel = enc.gumbel(i, k)
    return dist_nu, dist_gumbel
end
