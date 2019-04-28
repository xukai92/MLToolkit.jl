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

function (enc::MeanFieldSBC)(i, d::Int...)
    _eps = eps(FT)
    dist_kuma = enc.kuma(i, d...)
    dist_nu = BatchKumaraswamy(dist_kuma.a .+ _eps, dist_kuma.b .+ _eps)
    dist_gumbel = enc.gumbel(i, d...)
    return dist_nu, dist_gumbel
end

struct StructuredSBC <: StochasticLayer
    a
    b
    gumbel
end

"""
Helper function to initialise StructuredSBC with an IBP.
"""
function StructuredSBC(ibp::IBP{T}, i_dim::Int, k_init::Int=0; kwargs...) where {T<:AbstractFloat}
    return StructuredSBC(i_dim, k_init, ibp.α, one(T); kwargs...)
end

function StructuredSBC(i_dim::Int, k_init::Int=0, α::AbstractFloat=one(FT), β::AbstractFloat=one(FT), ; kwargs...)
    a = Knet.Param(AT(zeros(FT, k_init, 1) .+ invsoftplus(α)))
    b = Knet.Param(AT(zeros(FT, k_init, 1) .+ invsoftplus(β)))
    gumbel = GumbelBernoulliLogitLazyDense(i_dim, k_init; kwargs...)
    return StructuredSBC(a, b, gumbel)
end

function (enc::StructuredSBC)(i, d::Int...)
    _eps = eps(FT)
    # TODO: implement online initialisation
    # NOTE: for non-RR (i.e. static) version, d[1] is always equal to size(a, 1) and size(b, 1)
    dist_nu = BatchKumaraswamy(softplus.(enc.a[1:d[1],:]) .+ _eps,
                               softplus.(enc.b[1:d[1],:]) .+ _eps)
    dist_gumbel = enc.gumbel(i, d...)
    return dist_nu, dist_gumbel
end
