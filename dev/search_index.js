var documenterSearchIndex = {"docs":
[{"location":"plots/#MLToolkit.Plots-1","page":"MLToolkit.Plots","title":"MLToolkit.Plots","text":"","category":"section"},{"location":"plots/#","page":"MLToolkit.Plots","title":"MLToolkit.Plots","text":"CurrentModule = Plots","category":"page"},{"location":"plots/#","page":"MLToolkit.Plots","title":"MLToolkit.Plots","text":"Modules = [Plots]","category":"page"},{"location":"plots/#MLToolkit.Plots.FeatureActivations","page":"MLToolkit.Plots","title":"MLToolkit.Plots.FeatureActivations","text":"A plot of feature activations.\n\n\n\n\n\n","category":"type"},{"location":"plots/#MLToolkit.Plots.ImageGrid","page":"MLToolkit.Plots","title":"MLToolkit.Plots.ImageGrid","text":"A plot of images in grid.\n\n\n\n\n\n","category":"type"},{"location":"plots/#MLToolkit.Plots.LinesWithErrorBar","page":"MLToolkit.Plots","title":"MLToolkit.Plots.LinesWithErrorBar","text":"A plot of lines with shaded error bar.\n\n\n\n\n\n","category":"type"},{"location":"plots/#MLToolkit.Plots.OneDimFunction","page":"MLToolkit.Plots","title":"MLToolkit.Plots.OneDimFunction","text":"A plot of 1D function.\n\n\n\n\n\n","category":"type"},{"location":"plots/#MLToolkit.Plots.TwoDimFunction","page":"MLToolkit.Plots","title":"MLToolkit.Plots.TwoDimFunction","text":"A plot of 2D function.\n\n\n\n\n\n","category":"type"},{"location":"plots/#MLToolkit.Plots.TwoYAxesLines","page":"MLToolkit.Plots","title":"MLToolkit.Plots.TwoYAxesLines","text":"A plot of two lines with a shared y-axis.\n\n\n\n\n\n","category":"type"},{"location":"plots/#MLToolkit.Plots.get_tikz_code-Tuple{PyPlot.Figure,MLToolkit.Plots.AbstractPlot}","page":"MLToolkit.Plots","title":"MLToolkit.Plots.get_tikz_code","text":"get_tikz_code([fig], p::AbstractPlot; kwargs...)\n\nUsage:\n\nfig = plot(p)\ncode = get_tikz_code(fig, p)\n\n\n\n\n\n","category":"method"},{"location":"plots/#MLToolkit.Plots.make_imggrid-Tuple{AbstractArray{#s32,4} where #s32<:Number,Any,Any}","page":"MLToolkit.Plots","title":"MLToolkit.Plots.make_imggrid","text":"make_imggrid(img::AbstractArray{<:Number,4}, nrows, ncols; gap::Int=1)\n\nCreate a nrows by ncols image grid.\n\n\n\n\n\n","category":"method"},{"location":"plots/#MLToolkit.Plots.make_imggrid-Tuple{Any}","page":"MLToolkit.Plots","title":"MLToolkit.Plots.make_imggrid","text":"make_imggrid(img; kwargs...)\n\nCreate an approximately squared image grid based on the number of images. For example, if last(size(img)) is 100, the grid will be 10 by 10.\n\n\n\n\n\n","category":"method"},{"location":"plots/#MLToolkit.Plots.plot!-Tuple{MLToolkit.Plots.AbstractPlot,Vararg{Any,N} where N}","page":"MLToolkit.Plots","title":"MLToolkit.Plots.plot!","text":"plot!([ax=plt.gca()], p::AbstractPlot, args...; kwargs...)\n\nUsage:\n\nfig, ax = figure()\nplot!(ax, p)\n\n\n\n\n\n","category":"method"},{"location":"plots/#MLToolkit.Plots.plot-Tuple{MLToolkit.Plots.AbstractPlot,Vararg{Any,N} where N}","page":"MLToolkit.Plots","title":"MLToolkit.Plots.plot","text":"plot(p::AbstractPlot, args...; figsize=nothing, kwargs...)\n\nUsage:\n\nfig = plot(p)\n\n\n\n\n\n","category":"method"},{"location":"plots/#MLToolkit.Plots.savefig-Tuple{PyPlot.Figure,Union{Nothing, MLToolkit.Plots.AbstractPlot},String}","page":"MLToolkit.Plots","title":"MLToolkit.Plots.savefig","text":"savefig([fig], p::AbstractPlot, fname::String; bbox_inches=\"tight\", kwargs...)\n\nUsage:\n\nfig = plot(p)\nsavefig(fig, p, \"fig.png)\nsavefig(fig, p, \"fig.tex)\n\n\n\n\n\n","category":"method"},{"location":"#Welcome-to-the-Documentation-of-MLToolkit.jl-1","page":"Home","title":"Welcome to the Documentation of MLToolkit.jl","text":"","category":"section"},{"location":"#","page":"Home","title":"Home","text":"CurrentModule = MLToolkit","category":"page"}]
}