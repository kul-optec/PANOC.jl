using Test
using LinearAlgebra
using PANOC
using ProximalOperators

using Random

Random.seed!(0)

@testset "NonnegLS" begin
	T = Float64
    (m, k, n) = (100, 3, 100)
    A, B = randn(T, m, k), randn(T, m, n)
    f = Translate(SqrNormL2(), -B)
    g = IndNonnegative()

	@testset "Forward-backward envelope" begin
		Lf = opnorm(A)^2
		gamma = 0.95/Lf
		for X in [[zeros(T, k, n)]; [max.(0, randn(T, k, n)) for i=1:4]; [rand(T, k, n) for i=1:4]]
			grad_f_AX, f_AX = gradient(f, A*X)
			F_X = f_AX + g(X)
			At_grad_f_AX = A'grad_f_AX
			Z, g_Z = prox(g, X - gamma * At_grad_f_AX, gamma)
			res = X - Z
			FBE_X = PANOC.f_model(f_AX, At_grad_f_AX, res, gamma) + g_Z
			F_Z = f(A*Z) + g_Z
			tol = 10*eps(T)*(1+abs(FBE_X))
			@test T(0) <= F_X
			@test T(0) <= F_Z < T(Inf)
			@test T(0) <= FBE_X < T(Inf)
			@test FBE_X <= F_X - (0.5/gamma)*norm(res)^2 + tol
			@test F_Z <= FBE_X - (0.5/gamma)*(1 - gamma*Lf)*norm(res)^2 + tol
		end
	end

	@testset "PANOC(fixed)" begin
		for x0 in [[zeros(T, k, n)]; [max.(0, randn(T, k, n)) for i=1:3]; [rand(T, k, n) for i=1:3]]
			x_panoc, it_panoc = panoc(f, A, g, x0, L=opnorm(A)^2, maxit=1000, tol=1e-6, verbose=false)
			@test it_panoc <= 30
		end
	end

	@testset "PANOC(adaptive)" begin
		for x0 in [[zeros(T, k, n)]; [max.(0, randn(T, k, n)) for i=1:3]; [rand(T, k, n) for i=1:3]]
			x_panoc, it_panoc = panoc(f, A, g, x0, maxit=1000, tol=1e-6, verbose=false)
			@test it_panoc <= 30
		end
	end
end
