using MLToolkit, Test
using Distributions: Poisson

@testset "Distributions" begin
    @testset "DisplacedPoisson" begin
        for _ = 1:10
            λ = rand() * 10
            x = 0:1:100
            p = Poisson(λ)
            y_p = pdf.(Ref(p), x)
            m_p = mode(p)
            # When r equals 0, displaced Poisson reduces to normal Poisson.
            # Thus we can check the p.d.f using the Poisson (from Distributions.jl).
            dpeqv = DisplacedPoisson(λ, 0.0)
            y_dpeqv = pdf.(Ref(dpeqv), x)
            @test all(isapprox.(y_p, y_dpeqv))
            m_dpeqv = mode(dpeqv)
            @test m_p == m_dpeqv
        end
    end
    @testset "IBP" begin
        ibp = IBP(2.0)
        Z = rand(ibp, 10, 5)
        @test size(Z) == (5, 10)
    end
end
