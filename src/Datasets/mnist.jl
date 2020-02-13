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
    is_flatten::Bool=false,
    alpha::T=1f-6, 
    is_link::Bool=true,
    rng=MersenneTwister(seed),
) where {T<:AbstractFloat}
    if n_data > 60_000
        @warn "Attempts to access more than 60,000 train points in MNIST; clamped to 60,000."
        n_data = 60_000
    end
    if n_test > 10_000
        @warn "Attempts to access more than 10,000 test points in MNIST; clamped to 10,000."
        n_test = 10_000
    end
    X, y, Y, Xt, yt, Yt = get_image_data(
        MNIST, n_data, n_test, is_flatten, alpha, is_link, 10, (2, 1, 3); rng=rng
    )
    @info "Oh you just get the MNIST dataset" n_data=n_data n_test=n_test is_flatten=is_flatten alpha=alpha is_link=is_link
    return MNISTDataset{Val{is_link}, 784}(X, y, Y, Xt, yt, Yt)
end
