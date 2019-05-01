"""
    get_features_griffiths2011indian()

Generate the same features for the synthesised dataset used in (Griffiths and Ghahramani, 2011).

Ref: http://www.jmlr.org/papers/v12/griffiths11a.html
"""
function get_features_griffiths2011indian()
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