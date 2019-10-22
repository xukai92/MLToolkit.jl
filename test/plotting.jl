using Test, MLToolkit
using LaTeXStrings, Distributions

@testset "Plotting" begin
    @testset "make_imggrid" begin
        d = 5
        rand_img = rand(100, d^2)
        n_rows = n_cols = 3
        gap = 1
        imggrid = make_imggrid(rand_img, n_rows, n_cols)
        @test all(imggrid[1,:] .== 0.5)
        @test all(imggrid[end,:] .== 0.5)
        @test all(imggrid[:,1] .== 0.5)
        @test all(imggrid[:,end] .== 0.5)
        @test size(imggrid) == (n_rows * d + gap * (n_rows + 1),
                                n_cols * d + gap * (n_cols + 1))
    end

    @testset "plot and save" begin
        x = collect(1:0.1:10)
        y1 = sin.(x)
        y2 = x .^ 2

        p = plot(TwoYAxesLines(x=x, y1=y1, y2=y2, xlabel="x", ylabel1=L"\sin(x)", ylabel2=L"x^2"))

        save("test_two_y_axes_lines.tex", p, include_preamble=false)
        save("test_two_y_axes_lines.pdf", p)

        imgs = rand(10, 16, 16)

        p = plot(GrayImages(imgs))

        save("test_gray_images.tex", p, include_preamble=false)
        save("test_gray_images.pdf", p)

        p = plot(MvNormal(zeros(2), 1), (-3, 3), (-3, 3))

        save("test_contour_data.tex", p, include_preamble=false)
        save("test_contour_data.pdf", p)
    end
end
