using Test, MLToolkit, Knet
import AutoGrad

@testset "RR Rho" begin
    function obj(R, r::Rho; k_max=100)
        return sum([pdf(r, k) * R(k) for k = 1:k_max])
    end

    toyR1(i) = i <= 10 ? -i + 20 : 0.01i + 9.9
    toyR2(x) = (x - 5)^2 * 0.1

    for toyR in [toyR1, toyR2]

        toyR_min = min(toyR.(1:100)...)

        for m in [1, 5]

            pr = Param(Rho())
            pr.opt = SGD()

            for _ = 1:100_000*div(5, m)
                ϵ = randn(100) * 0.1
                dLdrho = MLToolkit.gradRR(x -> toyR(x) + ϵ[x], AutoGrad.value(pr), m)
                updategrad!(AutoGrad.value(pr), dLdrho)
                update!(AutoGrad.value(pr), pr.opt)
            end

            @test obj(toyR, AutoGrad.value(pr)) ≈ toyR_min atol=0.2*m

        end
    end
end
