for (node_sym, out_sym) in [(:GaussianNode, :Dense),
                            (:DynamicGaussianNode, :DynamicOut)]
    @eval begin
        struct $node_sym <: AbstractTrainable
            μ::AbstractTrainable
            Σ::AbstractTrainable
        end

        function $node_sym(i_dim::Integer, h_dim::Integer)
            return $node_sym($out_sym(i_dim, h_dim), $out_sym(i_dim, h_dim; f=softplus))
        end

        function (gn::$node_sym)(x, d::Integer...)
            return BatchNormal(gn.μ(x, d...), gn.Σ(x, d...))
        end
    end
end

for (node_sym, out_sym) in [(:GaussianLogVarNode, :Dense),
                            (:DynamicGaussianLogVarNode, :DynamicOut)]
    @eval begin
        struct $node_sym <: AbstractTrainable
            μ::AbstractTrainable
            logΣ::AbstractTrainable
        end

        function $node_sym(i_dim::Integer, z_dim::Integer)
            return $node_sym(out_sym(i_dim, z_dim), out_sym(i_dim, z_dim))
        end

        function (glvn::$node_sym)(x, d::Integer...)
            return BatchNormalLogVar(glvn.μ(x, d...), glvn.logΣ(x, d...))
        end
    end
end

for (node_sym, out_sym) in [(:BernoulliNode, :Dense),
                            (:DynamicBernoulliNode, :DynamicOut)]
    @eval begin
        struct $node_sym <: AbstractTrainable
            p::AbstractTrainable
        end

        function $node_sym(i_dim::Integer, z_dim::Integer)
            return $node_sym(out_sym(i_dim, z_dim; f=Knet.sigm))
        end

        function (bn::$node_sym)(x, d::Integer...)
            return BatchBernoulli(bn.p(x, d...))
        end
    end
end

for (node_sym, out_sym) in [(:BernoulliLogitNode, :Dense),
                            (:DynamicBernoulliLogitNode, :DynamicOut)]
    @eval begin
        struct $node_sym <: AbstractTrainable
            logitp::AbstractTrainable
        end

        function $node_sym(i_dim::Integer, z_dim::Integer)
            return $node_sym($out_sym(i_dim, z_dim))
        end

        function (bln::$node_sym)(x, d::Integer...)
            return BatchBernoulliLogit(bln.logitp(x, d...))
        end
    end
end
