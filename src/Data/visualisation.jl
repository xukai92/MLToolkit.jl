"""
    make_imggrid(x, n_rows, n_cols; flat=false, gap::Integer=1)

Transform `x` to a `n_rows` by `n_cols` grid.

Each instance in `x` is assumed to have a first dimension of `d^2` if flat,
or a first dimension of `d` if not flat.

NOTE: only gray imgs are supported.
"""
function make_imggrid(x, n_rows, n_cols; flat=true, gap::Integer=1)
    if !flat
        x = rehsape(x, size(x, 1)^2, size(x, 4))
    end
    d², n = size(x)
    d = round(Int, sqrt(d²))
    x_show = 0.5 * ones(FT, n_rows * (d + gap) + gap, n_cols * (d + gap) + gap)
    i = 1
    for row = 1:n_rows, col = 1:n_cols
        if i <= n
            row_i = (row - 1) * (d + gap) + 1
            col_i = (col - 1) * (d + gap) + 1
            x_show[row_i+1:row_i+d,col_i+1:col_i+d] = x[:,i]
        else
            break
        end
        i += 1
    end
    return x_show
end

function make_imggrid(x; kargs...)
    n = size(x, 2)
    l = ceil(Integer, sqrt(n))
    n_rows = l * (l - 1) > n ? l - 1 : l
    return make_imggrid(x, l, n_rows; kargs...)
end
