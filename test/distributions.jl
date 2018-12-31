using MLToolkit, Distributions, Test

@testset "DisplacedPoisson" begin
    x = 0:1:100
    y_p = Distributions.pdf.(Ref(Poisson(0.3)), x)
    y_dpeqv = pdf.(Ref(DisplacedPoisson(0.3, 0.0)), x)
    @test all(isapprox.(y_p, y_dpeqv))
end
