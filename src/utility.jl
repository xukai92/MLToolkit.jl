function count_leadingzeros(z::Vector{Int})
    N = length(z)
    n = 1
    # Iterate until the frist non-zero item
    while n <= N && z[n] == 0
        n = n + 1
    end
    return n - 1
end

turnoffgpu() = Knet.gpu(false)