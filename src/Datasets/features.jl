struct FeatureDataset{T, D} <: ImageDataset{T, D}
    X
    Xt
end

n_display(::FeatureDataset) = 64

function FeatureDataset(
    n_data::Int, 
    features::Matrix{T1}; 
    seed::Int=1,
    test_ratio=1/6,
    n_test::Int=ratio2num(n_data, test_ratio),
    alpha::T2=0f0,
    is_link::Bool=false,
) where {T1, T2}
    rng = MersenneTwister(seed)
    D, n_features = size(features)
    X = features * rand(rng, Bool, n_features, n_data)
    X = preprocess(rng, X, alpha, is_link)
    Xt = features * rand(rng, Bool, n_features, n_test)
    Xt = preprocess(rng, Xt, alpha, is_link)
    return FeatureDataset{Val{is_link}, D}(X, Xt)
end

"""
    get_features_griffiths2011()

Generate the same features for the synthesised dataset used in (Griffiths and Ghahramani, 2011).

Ref: http://www.jmlr.org/papers/v12/griffiths11a.html
"""
function get_features_griffiths2011()
    features = []

    push!(features, vec(Int[0 1 0 0 0 0;
                            1 1 1 0 0 0;
                            0 1 0 0 0 0;
                            0 0 0 0 0 0;
                            0 0 0 0 0 0;
                            0 0 0 0 0 0]))
                    
    push!(features, vec(Int[0 0 0 1 1 1;
                            0 0 0 1 0 1;
                            0 0 0 1 1 1;
                            0 0 0 0 0 0;
                            0 0 0 0 0 0;
                            0 0 0 0 0 0]))
                    
    push!(features, vec(Int[0 0 0 0 0 0;
                            0 0 0 0 0 0;
                            0 0 0 0 0 0;
                            1 0 0 0 0 0;
                            1 1 0 0 0 0;
                            1 1 1 0 0 0]))
                    
    push!(features, vec(Int[0 0 0 0 0 0;
                            0 0 0 0 0 0;
                            0 0 0 0 0 0;
                            0 0 0 1 1 1;
                            0 0 0 0 1 0;
                            0 0 0 0 1 0]))
                    
    features = hcat(features...)

    return features
end


"""
    get_features_xu2019()

Generate 35 12x12 features, the same features for the synthesised dataset used in (Xu et al., 2019).

Ref: http://proceedings.mlr.press/v97/xu19e.html
"""
function get_features_xu2019()

    features = []

    push!(features, vec(Int[0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;

                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0]))

    push!(features, vec(Int[1 0 0 0 0 0  0 0 0 0 0 0;
                            1 0 0 0 0 0  0 0 0 0 0 0;
                            1 1 1 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;

                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0]))

    push!(features, vec(Int[0 1 1 0 0 0  0 0 0 0 0 0;
                            0 1 1 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;

                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0]))

    push!(features, vec(Int[0 0 0 0 1 0  0 0 0 0 0 0;
                            0 0 0 1 1 1  0 0 0 0 0 0;
                            0 0 0 0 1 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;

                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0]))

    push!(features, vec(Int[0 0 0 1 0 1  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 1 0 1  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;

                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0]))

    push!(features, vec(Int[0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            1 1 1 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;

                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0]))

    push!(features, vec(Int[0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            1 1 1 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;

                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0]))

    push!(features, vec(Int[0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            1 1 1 0 0 0  0 0 0 0 0 0;

                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0]))

    push!(features, vec(Int[0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 1 1 1  0 0 0 0 0 0;
                            0 0 0 0 1 1  0 0 0 0 0 0;
                            0 0 0 0 0 1  0 0 0 0 0 0;

                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0]))

    push!(features, vec(Int[0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 1 0 0  0 0 0 0 0 0;
                            0 0 0 1 1 0  0 0 0 0 0 0;

                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0]))

    push!(features, vec(Int[0 0 0 0 0 0  1 1 1 0 0 0;
                            0 0 0 0 0 0  1 1 0 0 0 0;
                            0 0 0 0 0 0  1 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;

                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0]))

    push!(features, vec(Int[0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 1 0 0 0;
                            0 0 0 0 0 0  0 1 1 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;

                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0]))

    push!(features, vec(Int[0 0 0 0 0 0  0 0 0 1 1 1;
                            0 0 0 0 0 0  0 0 0 1 0 1;
                            0 0 0 0 0 0  0 0 0 1 1 1;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;

                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0]))

    push!(features, vec(Int[0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 1 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;

                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0]))

    push!(features, vec(Int[0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 1 0 0;
                            0 0 0 0 0 0  0 0 0 1 1 1;
                            0 0 0 0 0 0  0 0 0 1 0 0;

                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0]))

    push!(features, vec(Int[0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 1 1;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;

                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0]))

    push!(features, vec(Int[0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 1 1;

                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0]))

    push!(features, vec(Int[0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 1 0 0 0;
                            0 0 0 0 0 0  1 1 1 0 0 0;
                            0 0 0 0 0 0  0 0 1 0 0 0;

                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0]))

    push!(features, vec(Int[0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  1 1 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  1 1 0 0 0 0;

                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0]))

    push!(features, vec(Int[0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;

                            0 1 0 0 0 0  0 0 0 0 0 0;
                            0 1 0 0 0 0  0 0 0 0 0 0;
                            1 1 1 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0]))

    push!(features, vec(Int[0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;

                            1 0 1 0 0 0  0 0 0 0 0 0;
                            1 0 1 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0]))

    push!(features, vec(Int[0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;

                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            1 1 1 0 0 0  0 0 0 0 0 0;
                            0 1 0 0 0 0  0 0 0 0 0 0;
                            0 1 0 0 0 0  0 0 0 0 0 0]))

    push!(features, vec(Int[0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;

                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            1 0 1 0 0 0  0 0 0 0 0 0;
                            1 0 1 0 0 0  0 0 0 0 0 0]))

    push!(features, vec(Int[0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;

                            0 0 0 1 0 0  0 0 0 0 0 0;
                            0 0 0 1 1 0  0 0 0 0 0 0;
                            0 0 0 1 1 1  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0]))

    push!(features, vec(Int[0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;

                            0 0 0 0 1 1  0 0 0 0 0 0;
                            0 0 0 0 0 1  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0]))

    push!(features, vec(Int[0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;

                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 1  0 0 0 0 0 0;
                            0 0 0 0 1 1  0 0 0 0 0 0;
                            0 0 0 1 1 1  0 0 0 0 0 0]))

    push!(features, vec(Int[0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;

                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 1 1 0  0 0 0 0 0 0;
                            0 0 0 1 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0]))

    push!(features, vec(Int[0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;

                            0 0 0 0 0 0  1 1 0 0 0 0;
                            0 0 0 0 0 0  1 1 0 0 0 0;
                            0 0 0 0 0 0  1 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0]))

    push!(features, vec(Int[0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;

                            0 0 0 0 0 0  0 0 1 0 0 0;
                            0 0 0 0 0 0  0 0 1 0 0 0;
                            0 0 0 0 0 0  0 1 1 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0]))

    push!(features, vec(Int[0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;

                            0 0 0 0 0 0  0 0 0 1 1 1;
                            0 0 0 0 0 0  0 0 0 0 0 1;
                            0 0 0 0 0 0  0 0 0 1 1 1;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0]))

    push!(features, vec(Int[0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;

                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 1 1 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0]))

    push!(features, vec(Int[0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;

                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  1 0 1 1 1 1;
                            0 0 0 0 0 0  1 0 1 0 0 1;
                            0 0 0 0 0 0  1 1 1 0 0 1]))


    push!(features, vec(Int[0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;

                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 1 1 0;
                            0 0 0 0 0 0  0 0 0 1 1 0]))

    push!(features, vec(Int[0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;

                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0;
                            0 0 0 0 0 0  0 1 0 0 0 0;
                            0 0 0 0 0 0  0 1 0 0 0 0;
                            0 0 0 0 0 0  0 0 0 0 0 0]))

    features = hcat(features...)
    return features[:,[1,17,16,6,22,19,11,2,28,33,4,29,31,13,27,20,18,5,9,34,23,24,25,15,21,7,32,30,12,10,8,26,14,3]]
end
