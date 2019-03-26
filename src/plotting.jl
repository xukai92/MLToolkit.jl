function plot_two_y_axes(xs, ys1, ys2; color1="tab:red", color2="tab:blue",
                         xlabel=nothing, ylabel1=nothing, ylabel2=nothing)
    fig, ax1 = plt.subplots()
    ax1."plot"(xs, ys1, c=color1)
    xlabel == nothing || ax1."set_xlabel"(xlabel)
    ylabel1 == nothing || ax1."set_ylabel"(ylabel1, color=color1)
    ax1."tick_params"(axis="y", labelcolor=color1)
    ax2 = ax1["twinx"]()
    ax2."plot"(xs, ys2, c=color2)
    ylabel2 == nothing || ax2."set_ylabel"(ylabel2, color=color2)
    ax2."tick_params"(axis="y", labelcolor=color2)
    fig."tight_layout"()
    return fig
end
