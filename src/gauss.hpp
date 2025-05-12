#pragma once

#include <tuple>
#include <nd.hpp>
#include <ndarray.hpp>
#include <slice.hpp>
#include <primitives.hpp>

/// assumes [n,m] matrix and n <= m
template <class U>
void gaussSolve(U &&A) {
	auto [n, m] = A.shape();
	NDArray B((_, m), type<int>);
	for (int j = 0; j < std::min(n, m); ++j) {
		bool found = false;
		for (int i = j; i < n; ++i)
			if (A[i][j] == 1) {		// always is one because of rank N
				found = true;
				if (i == j) break;
				B	 = A[i];
				A[i] = A[j];
				A[j] = B;
				break;
			}
		if (!found) {}

		for (int i = 0; i < n; ++i) {
			if (i == j) continue;
			if (A[i][j] == 1) {
				A[i] -= A[j];
				A[i].apply([](int x) { return x & 1; });
			}
		}
	}
}

/// assumes [n,m] matrix and n <= m
template <class U>
void gaussSolveNonhomogenous(U &&A) {
	auto [n, m] = A.shape();
	NDArray B((_, m), type<int>);
	for (int j = 0; j < m-1; ++j) {
		bool found = false;
		for (int i = j; i < n; ++i)
			if (A[i][j] == 1) {		// always is one because of rank N
				found = true;
				if (i == j) break;
				B	 = A[i];
				A[i] = A[j];
				A[j] = B;
				break;
			}
		if (!found) {}

		for (int i = 0; i < n; ++i) {
			if (i == j) continue;
			if (A[i][j] == 1) {
				A[i] -= A[j];
				A[i].apply([](int x) { return x & 1; });
			}
		}
	}
}


template <class g>
auto orthogonal(g &&G) {
	using P		= std::pair<int, int>;
	auto [n, m] = G.shape();

	auto G_ = NDArray((_, n, m), type<int>);
	assign(G_, G_.shape(), G);

	gaussSolve(G_);

	NDArray H = NDArray((_, m - n, m), type<int>);
	auto	A = Slice(G_, G_.shape(), (_, P{0, n}, P{n, m}));
	auto	B = Slice(H, H.shape(), (_, P{0, m - n}, P{0, n}));
	assignTransposed(B, B.shape(), A);	   // -A^t

	Slice(H, H.shape(), (_, P{0, m - n}, P{n, m})) = Identity<int>(m - n);

	return H;
}

template <class g, class b>
auto solve(g &&G, b &&B) {
	using P = std::pair<int, int>;

	auto [n, m] = G.shape();
	auto [k]	= B.shape();
	//std::cout << std::format("G: [{}, {}] B: [{}] \n", n, m, k) << std::endl;
	assert(k == m);

	auto G_										 = NDArray((_, m, n+1), type<int>);
	assignTransposed(G_, (_, m, n), G);
	assignTransposed(
		Slice(G_, G_.shape(), (_, P{0, m}, P{n, n + 1})),
		(_, m, 1), 
		Repeat(B, 1, B.shape(), type<int>));

	gaussSolveNonhomogenous(G_);

	auto res = NDArray((_, n), type<int>);
	for(int i = 0; i < n; ++i) {
		res[i] = G_[i][n];
	}

	return res;
}
