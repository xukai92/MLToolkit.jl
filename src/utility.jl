function count_leadingzeros(z::Vector{Int})
    N = length(z)
    n = 1
    # Iterate until the frist non-zero item
    while n <= N && z[n] == 0
        n = n + 1
    end
    return n - 1
end

function include_list_as_module(list, module_name_prefix)
    return Distributed.map(list) do t
        @eval module $(Symbol("$(module_name_prefix)_", t))
            include($t)
        end
        return
    end
end

