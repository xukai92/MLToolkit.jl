"""
    load_mnist(mf::Function, tr_sz::Integer, te_sz::Integer; flatten::Bool=true)

Load a subset of MNIST-like dataset.
"""
function load_mnist(mf::Function, tr_sz::Integer, te_sz::Integer; flatten::Bool=true)
    X_tr, y_tr, X_te, y_te = mf()

    if flatten
        X_tr_sub = Matrix{FT}(undef, 28 * 28, tr_sz)
        X_te_sub = Matrix{FT}(undef, 28 * 28, te_sz)

        for i = 1:tr_sz
            X_tr_sub[:,i] = reshape(X_tr[:,:,1,i]', 28 * 28, 1)
        end

        for i = 1:te_sz
            X_te_sub[:,i] = reshape(X_te[:,:,1,i]', 28 * 28, 1)
        end
    else
        X_tr_sub = Array{FT,4}(X_tr[:,:,:,1:tr_sz])
        X_te_sub = Array{FT,4}(X_te[:,:,:,1:te_sz])
    end

    y_tr_sub = Vector{FT}(y_tr)[1:tr_sz]
    y_te_sub = Vector{FT}(y_te)[1:te_sz]

    return X_tr_sub, y_tr_sub, X_te_sub, y_te_sub
end
