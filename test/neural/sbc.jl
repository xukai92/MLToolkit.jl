using Test, MLToolkit

@testset "Neural SBC" begin
    i_dim = 20
    k_init = 10
    batch_size = 50

    @testset "$sbc" for sbc in [MeanFieldSBC(i_dim, k_init), StructuredSBC(i_dim, k_init)]
        is_all_true = true
        @test begin
            for k_out = 1:k_init
                x = AT(rand(FT, i_dim, batch_size))
                dist_kuma, dist_gumbel = sbc(x, k_out)
                is_all_true = is_all_true && dist_kuma isa BatchKumaraswamy
                is_all_true = is_all_true && dist_gumbel isa BatchGumbelBernoulliLogit
            end
            is_all_true
        end
    end

    @warn "`neural/sbc.jl` is not tested for online initialisation yet."
end
