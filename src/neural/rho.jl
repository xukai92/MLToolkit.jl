using LinearAlgebra: tril

mutable struct Rho{T}
    lnpd::LogitNPD{T,Vector{T}}
    grad::Vector{T}
end

Rho(; l_init=zero(FT)) = Rho(0; l_init=l_init)
Rho(alpha; l_init=zero(FT)) = Rho(ceil(Int, alpha); l_init=l_init)
Rho(k_init::Integer; l_init=zero(FT)) = Rho(LogitNPD(k_init; l_init=l_init), zeros(FT, k_init))

Base.size(r::Rho) = 0

getρ(r::Rho, args...) = getρ(r.lnpd, args...)
rand(r::Rho, args...) = rand(r.lnpd, args...)
pdf(r::Rho, args...) = pdf(r.lnpd, args...)

# TODO: implement a multi-sample version for the vectorized version below.
# TODO: make the function below into russian roulette
function gradRR(R_rr, rho_rr)
    M = (1 .- rho_rr) ./ rho_rr'
    M = tril(M)
    M[CartesianIndex.(Base.axes(M, 1), Base.axes(M, 1))] .= -1
    return vec(R_rr' * M)
end

function gradRR(R::Function, r::Rho)
    k_rr = rand(r)
    R_rr = R.(1:k_rr)
    rho_rr = getρ(r, 1, k_rr)
    return gradRR(R_rr, rho_rr)
end

function gradRR(R::Function, r::Rho, m::Integer)
    return gradRR(R::Function, r::Rho, rand(r.lnpd, m))
end

function gradRR(R::Function, r::Rho, k_list)
    local g_list = []
    for k_rr in k_list
        R_rr = R.(1:k_rr)
        rho_rr = getρ(r, 1, k_rr)
        push!(g_list, gradRR(R_rr, rho_rr))
    end
    g_list_length = map(g -> length(g), g_list)
    g_acc = zeros(max(g_list_length...))
    for (g, gl) in zip(g_list, g_list_length)
        g_acc[1:gl] .+= g
    end
    return g_acc / length(k_list)
end

# gradRR(R::Function, r::Rho, m::Int) = roll(R, r.lnpd, m)
# gradRR(R::Function, r::Rho, τs::AbstractVector{Int}) = roll(R, r.lnpd, τs)

function updategrad!(r, dLdrho::Vector{T}) where {T<:AbstractFloat}
    rho = getρ(r, 1, length(dLdrho))
    drhodlogitρ = rho .* (1 .- rho)
    dLdlogitρ = dLdrho .* drhodlogitρ
    llnpd = length(r.lnpd.logitρ)
    lgrad = length(dLdlogitρ)
    r.grad = llnpd > lgrad ? vcat(dLdlogitρ, zeros(T, llnpd - lgrad)) : dLdlogitρ
end

function update_by_dLdlogitρ!(pr::Knet.Param{Rho{T}}, dLdlogitρ) where {T}
    Knet.update!(value(pr), dLdlogitρ, pr.opt)
end

function Knet.update!(r::Rho, opt)
    opt = SGD()
    r.lnpd.logitρ .= Knet.update!(r.lnpd.logitρ, r.grad, opt)
end
Knet.update!(r::Rho, ::Nothing, opt) = update!(r, opt)

AutoGrad.grad(_, pr::Knet.Param{Rho{T}}) where {T} = nothing
