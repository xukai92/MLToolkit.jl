# NOTE: function starting with `make_` will return a figure instance while
#       function starting with `plot_` will return NOT

function make_two_y_axes_plot(xs, ys1, ys2; color1="tab:red", color2="tab:blue",
                              xlabel=nothing, ylabel1=nothing, ylabel2=nothing)
    fig, ax1 = plt.subplots()
    ax1."plot"(xs, ys1, c=color1)
    xlabel == nothing || ax1."set_xlabel"(xlabel)
    ylabel1 == nothing || ax1."set_ylabel"(ylabel1, color=color1)
    ax1."tick_params"(axis="y", labelcolor=color1)
    ax2 = ax1."twinx"()
    ax2."plot"(xs, ys2, c=color2)
    ylabel2 == nothing || ax2."set_ylabel"(ylabel2, color=color2)
    ax2."tick_params"(axis="y", labelcolor=color2)
    fig."tight_layout"()
    return fig
end

function plot_grayimg(img, args...; ax=plt.gca())
    @assert length(args) == 0 || length(args) == 2 "You can either plot a single image or declare the `n_rows` and `n_cols`"
    im = ax."imshow"(make_imggrid(img, args...), cmap="gray")
    plt.axis("off")
    divider = axes_grid1.make_axes_locatable(ax)
    cax = divider."append_axes"("bottom", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax, orientation="horizontal")
    return ax
end
