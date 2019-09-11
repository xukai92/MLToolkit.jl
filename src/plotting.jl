# NOTE: function starting with `make_xxx_plot` will return a figure instance while
#       function starting with `plot_xxx` will return the axes being plotted on

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

function plot_grayimg!(img, args...; ax=plt.gca())
    @assert length(args) == 0 || length(args) == 2 "You can either plot a single image or declare the `n_rows` and `n_cols`"
    im = ax."imshow"(make_imggrid(img, args...), cmap="gray")
    plt.axis("off")
    divider = axes_grid1.make_axes_locatable(ax)
    cax = divider."append_axes"("bottom", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax, orientation="horizontal")
    return ax
end

function plot_actmat!(Z::Matrix; ax=plt.gca())
    # TODO: implement a sorting version
    # col_sort_idcs = sortperm(vec([count_leadingzeros(Z[:,k]) for k = 1:size(Z, 2)]))
    # Z = Z[:,col_sort_idcs]
    ax."imshow"(Z, cmap="Greys", interpolation="nearest", vmin=0.0, vmax=1.0)
    ax.set_xticks([])
    ax.set_yticks([])
    return ax
end

function autoset_lim!(x; ax=plt.gca())
    xlims = [extrema(x[1,:])...]
    dx = xlims[2] - xlims[1]
    xlims += [-0.1dx, +0.1dx]
    ylims = [extrema(x[2,:])...]
    dy = ylims[2] - ylims[1]
    ylims += [-0.1dy, +0.1dy]
    dim = size(x, 1)
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
end

function plot_contour!(f; contourevals=100, alpha=0.3, ax=plt.gca())
    rangex = range(ax.get_xlim()..., length=contourevals)
    rangey = range(ax.get_ylim()..., length=contourevals)
    gridxy = [[xy...] for xy in Iterators.product(rangex, rangey)]
    gridx = map(xy -> xy[1], gridxy)
    gridy = map(xy -> xy[2], gridxy)
    gridz = f(hcat(gridxy[:]...))
    gridz = reshape(gridz, size(gridx)...)
    ax.contour(gridx, gridy, gridz, alpha=alpha)
end
