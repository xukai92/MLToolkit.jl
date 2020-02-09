using MLDatasets: MNIST

struct MNISTDataset{T, D} <: ImageDataset{T, D}
    X
    y
    Y
    Xt
    yt
    Yt
end

n_display(::MNISTDataset) = 100

function MNISTDataset(
    n_data::Int=60_000; 
    seed::Int=1,
    test_ratio=1/6,
    n_test::Int=ratio2num(n_data, test_ratio),
    is_flatten::Bool=true,
    alpha::T=0f0, 
    is_link::Bool=false,
) where {T}
    X, y, Y, Xt, yt, Yt = get_image_data(
        MNIST, n_data, n_test, is_flatten, alpha, is_link, 10, (2, 1, 3); seed=1
    )
    return MNISTDataset{Val{is_link}, 784}(X, y, Y, Xt, yt, Yt)
end
