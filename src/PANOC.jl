module PANOC

const RealOrComplex{R} = Union{R, Complex{R}}

const ArrayOrTuple{R} = Union{
	AbstractArray{C, N} where {C <: RealOrComplex{R}, N},
	Tuple{Vararg{AbstractArray{C, N} where {C <: RealOrComplex{R}, N}}}
}

const Maybe{T} = Union{T, Nothing}

using ProximalOperators
using Printf

include("utils/LBFGS.jl")

struct PANOC_iterable{R <: Real}
    f   # smooth term
    A   # matrix/linear operator
    g   # (possibly) nonsmooth, proximable term
    x0::ArrayOrTuple{R}  # initial point
    alpha::R   # in (0, 1), e.g.: 0.95
    beta::R   # in (0, 1), e.g.: 0.5
	L::Maybe{R}	# Lipschitz constant of the gradient of x ↦ f(Ax)
	memory	# memory parameter for L-BFGS
end

mutable struct PANOC_state{R <: Real}
    x   # iterate
    Ax  # A times x
    f_Ax::R   # value of smooth term
	grad_f_Ax	# gradient of f at Ax
    At_grad_f_Ax    # gradient of smooth term
    gamma::R   # stepsize
	y	# forward point
    z   # forward-backward point
    g_z::R # value of nonsmooth term (at z)
    res # fixed-point residual at iterate (= z - x)
    H::LBFGS{R}   # variable metric
    tau::Maybe{R} # stepsize (can be nothing since the initial state doesn't have it)
end

f_model(f_x, grad_f_x, res, gamma) = f_x - real(dot(grad_f_x, res)) + (0.5/gamma)*norm(res)^2

f_model(state::PANOC_state) = f_model(state.f_Ax, state.At_grad_f_Ax, state.res, state.gamma)

import Base: iterate

function iterate(iter::PANOC_iterable{R}) where R
    x = iter.x0
    Ax = iter.A * x
    grad_f_Ax, f_Ax = gradient(iter.f, Ax)

	L = iter.L
	if L === nothing
		# compute lower bound to Lipschitz constant of the gradient of x ↦ f(Ax)
	    xeps = x .+ one(R)
	    grad_f_Axeps, f_Axeps = gradient(iter.f, iter.A*xeps)
	    L = norm(iter.A' * (grad_f_Axeps - grad_f_Ax)) / sqrt(length(x))
	end

    gamma = iter.alpha/L

    # compute initial forward-backward step
    At_grad_f_Ax = iter.A' * grad_f_Ax
    y = x - gamma .* At_grad_f_Ax
    z, g_z = prox(iter.g, y, gamma)

    # compute initial fixed-point residual
    res = x - z

    # initialize variable metric
    H = LBFGS(x, iter.memory)

    state = PANOC_state{R}(x, Ax, f_Ax, grad_f_Ax, At_grad_f_Ax, gamma, y, z, g_z, res, H, nothing)

    return state, state
end

function iterate(iter::PANOC_iterable{R}, state::PANOC_state{R}) where R
	f_upp_Az = f_model(state)

    # backtrack gamma (warn and halt if gamma gets too small)
    while iter.L === nothing
        if state.gamma < 1e-7 # TODO: make this a parameter?
            @warn "parameter `gamma` became too small, stopping the iterations"
            return nothing
        end
        grad_f_Az, f_Az = gradient(iter.f, iter.A*state.z)
		tol = 10*eps(R)*(1 + abs(f_Az))
        if f_Az <= f_upp_Az + tol break end
        state.gamma *= 0.5
        state.y .= state.x .- state.gamma .* state.At_grad_f_Ax
        state.g_z = prox!(state.z, iter.g, state.y, state.gamma)
        state.res .= state.x .- state.z
        reset!(state.H)
        f_upp_Az = f_model(state)
    end

    # compute FBE
    FBE_x = f_upp_Az + state.g_z

    # update metric
    update!(state.H, state.x, state.res)

    # compute direction
    d = -(state.H*state.res)

    # backtrack tau 1 → 0
    Ad = iter.A * d
    Ares = zero(Ad)
    tau = one(R)
    x_new = state.x + d
    Ax_new = state.Ax + Ad

	sigma = iter.beta * (0.5/state.gamma) * (1 - iter.alpha)

    for i = 1:10
		# TODO write the next directly on the state (for efficiency)
        grad_f_Ax_new, f_Ax_new = gradient(iter.f, Ax_new)
        At_grad_f_Ax_new = iter.A' * grad_f_Ax_new
        y_new = x_new - state.gamma * At_grad_f_Ax_new
        z_new, g_z_new = prox(iter.g, y_new, state.gamma)
        res_new = x_new - z_new
        FBE_x_new = f_model(f_Ax_new, At_grad_f_Ax_new, res_new, state.gamma) + g_z_new

		tol = 10*eps(R)*(1 + abs(FBE_x))
        if FBE_x_new <= FBE_x - sigma * norm(state.res)^2 + tol
			# TODO write only what's left
            state.x = x_new
            state.Ax = Ax_new
			state.f_Ax = f_Ax_new
			state.grad_f_Ax = grad_f_Ax_new
            state.At_grad_f_Ax = At_grad_f_Ax_new
			state.y = y_new
            state.z = z_new
            state.g_z = g_z_new
            state.res = res_new
            state.tau = tau
            return state, state
        end

        if tau == one(R)
            Ares = iter.A * state.res
        end

        tau *= 0.5
        x_new .= state.x .+ tau .* d .- (1-tau) .* state.res
        Ax_new .= state.Ax .+ tau .* Ad .- (1-tau) .* Ares
    end

    @warn "stepsize `tau` became too small, stopping the iterations"
    return nothing
end

using Base.Iterators
include("utils/IterativeMethodsTools.jl")

export panoc

function panoc(f, A, g, x0;
	alpha=0.95, beta=0.5, L=nothing, memory=5,
	maxit=1000, tol=1e-8,
	verbose=true, freq=10)

	stop(state::PANOC_state) = norm(state.res)/state.gamma <= tol
	disp((it, state)) = @printf "%5d | %.3e | %.3e | %.3e\n" it state.gamma norm(state.res)/state.gamma (state.tau === nothing ? 0.0 : state.tau)

	iter = PANOC_iterable(f, A, g, x0, alpha, beta, L, memory)
	iter = take(halt(iter, stop), maxit)
	iter = enumerate(iter)
	if verbose iter = tee(sample(iter, freq), disp) end

	num_iters, state_final = loop(iter)

	return state_final.z, num_iters
end

end # module
