using Test, MLToolkit

@testset "Nonparametric discrete distribution" begin
    lnpd = LogitNPD()

    k_init = 100
    lnpd1 = LogitNPD(k_init)
    @test length(lnpd1.logitρ) == k_init
    @test lnpd1.logitρ == zeros(k_init)
    @test getlogitρ(lnpd1, 1, k_init) == zeros(k_init)
    @test getρ(lnpd1, 1, k_init) == ones(k_init) / 2
    @test pdf(lnpd1, 0) == 0
    @test pdf(lnpd1, 1) == 0.5
    @test pdf(lnpd1, 2) == 0.25
    @test logpdf(lnpd1, 1) == -log(FT(2))
    @test cdf(lnpd1, 0) == 0
    @test cdf(lnpd1, 1) == 0.5
    @test cdf(lnpd1, 2) == 0.75
    @test invlogcdf(lnpd1, log(FT(0.99))) == 7
    @test ccdf(lnpd1, 0) == 1
    @test ccdf(lnpd1, 1) == 0.5
    @test ccdf(lnpd1, 2) == 0.25
    @test mode(lnpd1) == 1

    k_init = 100
    lnpd1 = LogitNPD(k_init; l_init=1.0)
    @test lnpd1.logitρ == ones(k_init)
    @test getlogitρ(lnpd1, 1, k_init) == ones(k_init)
end
