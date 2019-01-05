using MLToolkit, Test
using Distributions: Beta

dims = (10, 5)

@testset "Activations" begin
    @testset "break_stick_ibp" begin
        ν = ones(2, 3) ./ 2
        π = break_stick_ibp(ν)
        @test π ≈ [0.5 0.5 0.5; 0.25 0.25 0.25]
    end
    @testset "break_logstick_ibp" begin
        for _ = 1:10
            ν = rand(Beta(2, 1), dims...)
            logν = log.(ν)
            @test all(log.(break_stick_ibp(ν)) .≈ break_logstick_ibp(logν))
        end
    end
end
