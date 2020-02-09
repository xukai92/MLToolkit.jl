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
    is_flatten::Bool=true,
    alpha::T=0f0, 
    is_link::Bool=false,
) where {T}
    X, y, Y, Xt, yt, Yt = get_image_data(
        CIFAR10, n_data, n_test, is_flatten, alpha, is_link, 10, (2, 1, 3, 4); seed=1
    )
    return CIFAR10Dataset{Val{is_link}, 3072}(X, y, Y, Xt, yt, Yt)
end
