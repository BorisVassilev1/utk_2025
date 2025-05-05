#pragma once
#include "hadamard.hpp"
#include "ndarray.hpp"
#include "primitives.hpp"
#include "slice.hpp"

inline auto Golay24() {
	using P = std::pair<int, int>;

	auto G = Ones((_, 12, 24), type<int>);
	
	Slice(G, G.shape(), (_, P{0, 12}, P{ 0, 12})) = Identity<int>(12); //.print(std::cout);
	
	auto A = hadamardPaley(12);
	matrixToCode(A);

	auto U = Slice(G, G.shape(), (_, P{1, 12}, P{12, 23}));
	auto V = Slice(A, A.shape(), (_, P{1, 12}, P{1, 12}));
	V.apply([](int x) { return !x; });
	//std::cout << "U" << U.shape() << " V" << V.shape() << std::endl;
	assignTransposed(U, U.shape(), V);
	G[0][23] = 0;
	

	return G;
}

template<class A, class B>
int dot(A && a, B && b) {
	int res = 0;
	for(int i = 0; i < std::get<0>(a.shape()); ++i) {
		res += a[i] * b[i];
	}
	return res;
}

template< class A>
int w(A && a) {
	int res = 0;
	for(int i = 0; i < std::get<0>(a.shape()); ++i) {
		res += a[i];
	}
	return res;
}

template<class V, class G>
inline bool isSolution(V && v, G && g) {
	for(int i = 0; i < std::get<0>(g.shape()); ++i) {
		if(dot(g[i], v) % 2 != 0) {
			return false;
		}
	}
	return true;
}

template<class V>
inline bool isSelfOrthogonal(V && v) {
	for(int i = 0; i < std::get<0>(v.shape()); ++i) {
		if(!isSolution(v[i], v)) {
			return false;
		}
	}
	return true;
}

template<class G>
inline int findDistance(G && g) {
	int min = std::get<0>(g.shape());
	for(int i = 0; i < std::get<0>(g.shape()); ++i) {
		int d = w(g[i]);
		min = std::min(min, d);
	}
	return min;
}

template<class G>
inline auto codeInfo(G && g) {
	int d = findDistance(g);
	auto [n, m] = g.shape();
	
	return std::make_tuple(m, n, d);
}

template<class G>
void printCodeInfo(G && g, const char* name) {
	auto [n, m, k] = codeInfo(std::forward<G>(g));
	std::cout << name << ": [" << n << ", " << m << ", " << k << "]" << std::endl;
}

#define CODE_INFO(g) printCodeInfo(g, #g);


