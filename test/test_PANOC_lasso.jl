using Test
using LinearAlgebra
using PANOC
using ProximalOperators
using RecursiveArrayTools: ArrayPartition, unpack

using Random

Random.seed!(0)

@testset "Lasso (tiny)" begin
	T = Float64
	A = T[
		1.0  -2.0   3.0  -4.0  5.0;
		2.0  -1.0   0.0  -1.0  3.0;
		-1.0   0.0   4.0  -3.0  2.0;
		-1.0  -1.0  -1.0   1.0  3.0
	]
	b = T[1.0, 2.0, 3.0, 4.0]
	m, n = size(A)
	f = Translate(SqrNormL2(), -b)
	lam = 0.1*norm(A'*b, Inf)
	g = NormL1(lam)
	Lf = opnorm(A)^2
	x_star = T[-3.877278911564627e-01, 0, 0, 2.174149659863943e-02, 6.168435374149660e-01]

	@testset "Forward-backward envelope" begin
		gamma = 0.95/Lf
		for x in [[zeros(T, n)]; [randn(T, n) for k=1:4]]
			grad_f_Ax, f_Ax = gradient(f, A*x)
			F_x = f_Ax + g(x)
			At_grad_f_Ax = A'grad_f_Ax
			z, g_z = prox(g, x - gamma * At_grad_f_Ax, gamma)
			res = x - z
			FBE_x = PANOC.f_model(f_Ax, At_grad_f_Ax, res, gamma) + g_z
			F_z = f(A*z) + g_z
			tol = 10*eps(T)*(1+abs(FBE_x))
			@test T(0) <= F_x < T(Inf)
			@test T(0) <= F_z < T(Inf)
			@test T(0) <= FBE_x < T(Inf)
			@test FBE_x <= F_x - (0.5/gamma)*norm(res)^2 + tol
			@test F_z <= FBE_x - (0.5/gamma)*(1 - gamma*Lf)*norm(res)^2 + tol
		end
	end

	@testset "PANOC (fixed)" begin
		for x0 in [[zeros(T, n)]; [randn(T, n) for k=1:4]]
			x_panoc, it_panoc = panoc(f, A, g, x0, L=Lf, maxit=1000, tol=1e-8, verbose=false)
			@test x_panoc ≈ x_star
			@test it_panoc <= 50
		end
	end

	@testset "PANOC (adaptive)" for verb=[false, true]
		for x0 in [[zeros(T, n)]; [randn(T, n) for k=1:4]]
			x_panoc, it_panoc = panoc(f, A, g, x0, maxit=1000, tol=1e-8, verbose=verb)
			@test x_panoc ≈ x_star
			@test it_panoc <= 50
		end
	end

	@testset "PANOC (with ArrayPartition, take 1)" begin
		f_alt = LeastSquares(A, b)
		f_twice = SeparableSum(f_alt, f_alt)
		g_twice = SeparableSum(g, g)

		for x0 in [[zeros(T, n)]; [randn(T, n) for k=1:4]]
			x0_twice = ArrayPartition(x0, x0)
			x_panoc, it_panoc = panoc(f_twice, I, g_twice, x0_twice, maxit=1000, tol=1e-8, verbose=false)
			@test unpack(x_panoc, 1) ≈ x_star
			@test unpack(x_panoc, 2) ≈ x_star
			@test it_panoc <= 50
		end
	end

end

@testset "Lasso (medium)" begin
	T = Float64
	A, b = randn(T, 200, 1000), randn(T, 200)
	m, n = size(A)
	f = Translate(SqrNormL2(), -b)
	lam = 0.1*norm(A'*b, Inf)
	g = NormL1(lam)
	Lf = opnorm(A)^2

	function fixed_point_residual(x, A, b, lam)
		gam = 1.0/opnorm(A)^2
		y = x - gam*(A'*(A*x - b))
		z = sign.(y) .* max.(abs.(y) .- gam*lam, 0.0)
		return z - x
	end

	@testset "PANOC(fixed)" begin
		for x0 in [[zeros(T, n)]; [randn(T, n) for k=1:4]]
			x_panoc, it_panoc = panoc(f, A, g, x0, L=Lf, maxit=1000, tol=1e-8, verbose=false)
			@test norm(fixed_point_residual(x_panoc, A, b, lam), Inf) <= 1e-8
			@test it_panoc < 1000
		end
	end

	@testset "PANOC(adaptive)" begin
		for x0 in [[zeros(T, n)]; [randn(T, n) for k=1:4]]
			x_panoc, it_panoc = panoc(f, A, g, x0, maxit=1000, tol=1e-8, verbose=false)
			@test norm(fixed_point_residual(x_panoc, A, b, lam), Inf) <= 1e-8
			@test it_panoc < 1000
		end
	end
end
