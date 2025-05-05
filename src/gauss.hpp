#pragma once

#include <tuple>
#include <nd.hpp>
#include <ndarray.hpp>

template<class U>
void gaussSolve(U && A) {
	auto [n, m] = A.shape();
	NDArray B((_, m), type<int>);
	for(int j = 0; j < n; ++j) {
		bool found = false;
		for(int i = j; i < n; ++i)
			if(A[i][j] == 1) { // always is one because of rank N
				found = true;
				if(i == j) break;
				B = A[i];
				A[i] = A[j];
				A[j] = B;
				break;
			}
		if(!found) {
			
		}

		for(int i = 0; i < n; ++i) {
			if(i == j) continue;
			if(A[i][j] == 1) {
				A[i] -= A[j];
				A[i].apply([](int x) { return x & 1; });
			}
		}

	}
}

template <class g>
auto orthogonal(g && G) {
	using P = std::pair<int, int>;
	auto [n, m] = G.shape();

	gaussSolve(G);

	NDArray H = NDArray((_, m-n, m), type<int>);
	auto A = Slice(G, G.shape(), (_, P{0, n}, P{n, m}));
	auto B = Slice(H, H.shape(), (_, P{0, m-n}, P{0, n}));
	assignTransposed(B, B.shape(), A); // -A^t
	
	Slice(H, H.shape(), (_, P{0, m-n}, P{n, m})) = Identity<int>(m-n);

	return H;
}


