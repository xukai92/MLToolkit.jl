using MLToolkit, Distributions, Test

@testset "DisplacedPoisson" begin
    for _ = 1:10
        λ = rand() * 10
        x = 0:1:100
        y_p = Distributions.pdf.(Ref(Poisson(λ)), x)
        # When r equals 0, displaced Poisson reduces to normal Poisson.
        # Thus we can check the p.d.f using the Poisson (from Distributions.jl).
        y_dpeqv = pdf.(Ref(DisplacedPoisson(λ, 0.0)), x)
        @test all(isapprox.(y_p, y_dpeqv))
    end
end
