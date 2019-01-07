using Test, MLToolkit
using Distributions: Poisson

@testset "DisplacedPoisson" begin
    for _ = 1:NUM_RANDTESTS
        λ = rand() * 10

        p = Poisson(λ)

        x = 0:1:100
        y_p = pdf.(Ref(p), x)
        # When r equals 0, displaced Poisson reduces to normal Poisson.
        # Thus we can check the p.d.f using the Poisson (from Distributions.jl).
        dpeqv = DisplacedPoisson(λ, 0.0)
        y_dpeqv = pdf.(Ref(dpeqv), x)
        @test all(isapprox.(y_p, y_dpeqv))

        m_p = mode(p)
        m_dpeqv = mode(dpeqv)
        @test m_p == m_dpeqv
    end
end
