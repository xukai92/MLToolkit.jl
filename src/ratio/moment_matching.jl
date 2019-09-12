pairwise_dot(x) = pairwise(SqEuclidean(), x; dims=2)

function pairwise_dot_kai(x)
    n = size(x, 2)
    xixj = x' * x
    xsq = sum(x .^ 2; dims=1)
    return repeat(xsq'; outer=(1, n)) + repeat(xsq; outer=(n, 1)) - 2xixj
end

# pairwise_dot(x::CuArray) = pairwise_dot_kai(x)

pairwise_dot(x, y) = pairwise(SqEuclidean(), x, y; dims=2)

function pairwise_dot_kai(x, y)
    nx = size(x, 2)
    ny = size(y, 2)
    xiyj = x' * y
    xsq = sum(x .^ 2; dims=1)
    ysq = sum(y .^ 2; dims=1)
    return repeat(xsq'; outer=(1, ny)) .+ repeat(ysq; outer=(nx, 1)) - 2xiyj
end

# pairwise_dot(x::CuArray, y::CuArray) = pairwise_dot_kai(x, y)

gaussian_gram_by_pairwise_dot(pdot; σ=1) = exp.(-pdot ./ 2(σ ^ 2))
gaussian_gram(x; σ=1) = gaussian_gram_by_pairwise_dot(pairwise_dot(x); σ=σ)
gaussian_gram(x, y; σ=1) = gaussian_gram_by_pairwise_dot(pairwise_dot(x, y); σ=σ)

function _estimate_r_de(pdot_dede, pdot_denu, get_r_hat, σ; kwargs...)
    Kdede = gaussian_gram_by_pairwise_dot(pdot_dede; σ=σ)
    Kdenu = gaussian_gram_by_pairwise_dot(pdot_denu; σ=σ)
    return get_r_hat(Kdede, Kdenu; kwargs...)
end

function estimate_r_de(x_de, x_nu; get_r_hat=get_r_hat_analytical, σs=nothing, kwargs...)
    pdot_dede = pairwise_dot_kai(x_de)
    pdot_denu = pairwise_dot_kai(x_de, x_nu)

    if isnothing(σs)
        σ = sqrt(median([pdot_dede..., pdot_denu...]))
        @info "Automatically choose σ using the square root of the median of pairwise distances: $σ."
        σs = [σ]
    end

    r_de = _estimate_r_de(pdot_dede, pdot_denu, get_r_hat, σs[1]; kwargs...)
    for σ in σs[2:end]
        r_de += _estimate_r_de(pdot_dede, pdot_denu, get_r_hat, σ; kwargs...)
    end
    
    return r_de ./ length(σs)
end

function get_r_hat_numerically(Kdede, Kdenu; positive=true, normalisation=true)
    n_de, n_nu = size(Kdenu)
    model = JuMP.Model(JuMP.with_optimizer(Ipopt.Optimizer; print_level=0))
    JuMP.@variable(model, r[1:n_de])
    JuMP.@objective(model, Min, 1 / n_de ^ 2 * sum(r[i] * Kdede[i,j] * r[j] for i = 1:n_de, j=1:n_de) - 2 / (n_de * n_nu) * sum(r[i] * Kdenu[i,j] for i = 1:n_de, j=1:n_nu))
    if positive
        JuMP.@constraint(model, r .>= 0)
    end
    if normalisation
        JuMP.@constraint(model, 1 / n_de * sum(r) == 1)
    end
    JuMP.optimize!(model)
    return JuMP.value.(r)
end

function get_r_hat_analytical(Kdede, Kdenu; T=eltype(Kdede), ϵ=convert(T, 1 / 1_000))
    n_de, n_nu = size(Kdenu)
    return T(n_de / n_nu) * inv(Kdede + diagm(0 => ϵ * ones(Float32, n_de))) * Kdenu * ones(T, n_nu) 
end
