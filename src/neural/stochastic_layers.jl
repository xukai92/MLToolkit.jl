const STATIC_SYM_LIST = [:Dense, :LazyDense, :DynamicIn, :DynamicOut]

const UNARY_DIST_DICT = Dict(
    :Bernoulli => (:BatchBernoulli, :p, :(Knet.sigm)),
    :BernoulliLogit => (:BatchBernoulliLogit, :logitp, :identity),
    :GumbelBernoulliLogit => (:BatchGumbelBernoulliLogit, :logitp, :identity),
    :GumbelSoftmax => (:BatchGumbelSoftmax, :p, :identity),
)

for dist_sym in keys(UNARY_DIST_DICT)
    (batch_dist_sym, field_sym, f_sym) = UNARY_DIST_DICT[dist_sym]
    for static_sym in STATIC_SYM_LIST
        sto_sym = Symbol("$dist_sym$static_sym")

        @eval begin
            struct $sto_sym <: StochasticLayer
                $field_sym::StaticLayer
            end

            function $sto_sym(i_dim::Int, z_dim::Int; kwargs...)
                return $sto_sym($static_sym(i_dim, z_dim; f=$f_sym, kwargs...))
            end

            function (sto::$sto_sym)(x, d::Int...)
                return $batch_dist_sym(sto.$field_sym(x, d...))
            end
        end
    end
end

const BINARY_DIST_DICT = Dict(
    :Gaussian => (:BatchNormal, :μ, :identity,
                                :Σ, :softplus),
    :GaussianLogVar => (:BatchNormalLogVar, :μ, :identity,
                                            :logΣ, :identity),
    :Kumaraswamy => (:BatchKumaraswamy, :a, :softplus_safe,
                                        :b, :softplus_safe),
)

for dist_sym in keys(BINARY_DIST_DICT)
    (batch_dist_sym, field_sym_1, f_sym_1,
                     field_sym_2, f_sym_2) = BINARY_DIST_DICT[dist_sym]
    for static_sym in STATIC_SYM_LIST
        sto_sym = Symbol("$dist_sym$static_sym")

        @eval begin
            struct $sto_sym <: StochasticLayer
                $field_sym_1::StaticLayer
                $field_sym_2::StaticLayer
            end

            function $sto_sym(i_dim::Int, z_dim::Int; kwargs...)
                return $sto_sym($static_sym(i_dim, z_dim; f=$f_sym_1, kwargs...),
                                $static_sym(i_dim, z_dim; f=$f_sym_2, kwargs...))
            end

            function (sto::$sto_sym)(x, d::Int...)
                return $batch_dist_sym(sto.$field_sym_1(x, d...),
                                       sto.$field_sym_2(x, d...))
            end
        end
    end
end
