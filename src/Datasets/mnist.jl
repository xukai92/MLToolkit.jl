using MLDatasets: MNIST
using MLDataUtils: convertlabel, LabelEnc

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
    n_data::Int; 
    seed::Int=1,
    test_ratio=1/6,
    n_test::Int=ratio2num(n_data, test_ratio),
    is_flatten::Bool=true,
    alpha::T=0f0, 
    is_link::Bool=false,
) where {T}
    rng = MersenneTwister(seed)
    onehot_enc = LabelEnc.OneOfK(10)

    function process(X, y)
        if is_flatten
            X = reshape(permutedims(X, (2, 1, 3)), 784, :)
        end
        X = preprocess(rng, X, alpha, is_link)
        Y = convertlabel(onehot_enc, y)
        return X, y, Y
    end

    X = MNIST.traintensor(T, 1:n_data)
    y = MNIST.trainlabels(1:n_data) .+ 1
    X, y, Y = process(X,  y)

    Xt = MNIST.testtensor(T, 1:n_test)
    yt = MNIST.testlabels(1:n_test) .+ 1
    Xt, yt, Yt = process(Xt, yt)
    
    return MNISTDataset{Val{is_link}, 784}(X, y, Y, Xt, yt, Yt)
end
