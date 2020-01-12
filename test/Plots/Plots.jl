using Test, LaTeXStrings, Distributions
using MLToolkit.Plots

@testset "make_imggrid" begin
    d = 5
    rand_img = rand(d^2, 100)
    n_rows = n_cols = 3
    gap = 1
    imggrid = make_imggrid(rand_img, n_rows, n_cols)
    @test all(imggrid[1,:] .== 0.5)
    @test all(imggrid[end,:] .== 0.5)
    @test all(imggrid[:,1] .== 0.5)
    @test all(imggrid[:,end] .== 0.5)
    @test size(imggrid) == (
        n_rows * d + gap * (n_rows + 1),
        n_cols * d + gap * (n_cols + 1)
    )
end

@testset "plot and save" begin
    # Two axes lines
    x = collect(1:0.1:10)
    y1 = sin.(x)
    y2 = x .^ 2

    p = TwoYAxesLines(x, y1, y2)
    fig = plot(p, "--"; xlabel="x", ylabel1=L"\sin(x)", ylabel2=L"x^2")

    savefig(fig, p, "test.tex")
    savefig(fig, p, "test.png"; bbox_inches="tight")

    # # Images
    # imgs = GrayImages(rand(16, 16, 10))

    # plt.figure()
    # plot!(imgs)

    # plt.savefig("test_gray_images.pdf", bbox_inches="tight")


    # # Two-dimensional density
    # p = plot(MvNormal(zeros(2), 1), (-3, 3), (-3, 3))

    # save("test_contour_data.tex", p, include_preamble=false)
    # save("test_contour_data.pdf", p)
end