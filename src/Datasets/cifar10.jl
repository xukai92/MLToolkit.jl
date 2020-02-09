using MLDatasets: CIFAR10

struct CIFAR10Dataset{T, D} <: ImageDataset{T, D}
    X
    y
    Y
    Xt
    yt
    Yt
end

n_display(::CIFAR10Dataset) = 100

function CIFAR10Dataset(
    n_data::Int=50_000; 
    seed::Int=1,
    test_ratio=1/5,
    n_test::Int=ratio2num(n_data, test_ratio),
    is_flatten::Bool=false,
    alpha::T=5f-2, 
    is_link::Bool=true,
    rng=MersenneTwister(seed),
) where {T<:AbstractFloat}
    if n_data > 50_000
        @warn "Attempts to access more than 50,000 train points in MNIST; clamped to 50,000."
        n_data = 50_000
    end
    if n_test > 10_000
        @warn "Attempts to access more than 10,000 test points in MNIST; clamped to 10,000."
        n_test = 10_000
    end
    X, y, Y, Xt, yt, Yt = get_image_data(
        CIFAR10, n_data, n_test, is_flatten, alpha, is_link, 10, (2, 1, 3, 4); rng=rng
    )
    return CIFAR10Dataset{Val{is_link}, 3072}(X, y, Y, Xt, yt, Yt)
end
