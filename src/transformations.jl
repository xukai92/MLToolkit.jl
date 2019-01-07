"""
    break_stick_ibp(ν)

Break stick (IBP).
"""
function break_stick_ibp(ν)
    Kmax = size(ν, 1)
    p = []
    push!(p, ν[1,:])
    for k = 2:Kmax
        push!(p, p[k-1] .* ν[k,:])
    end
    p = hcat(p...)'
    return p
end

"""
    break_logstick_ibp(logν)

Break stick (IBP) in logorithmic space.
"""
function break_logstick_ibp(logν)
    Kmax = size(logν, 1)
    logp = []
    push!(logp, logν[1,:])
    for k = 2:Kmax
        push!(logp, logp[k-1] .+ logν[k,:])
    end
    logp = hcat(logp...)'
    return logp
end
