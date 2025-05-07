#pragma once

#include <ndarray.hpp>
#include <primitives.hpp>
#include <slice.hpp>
#include <prime.hpp>

inline auto hadamardPaley(int n) {
	using P = std::pair<int, int>;

	int p = n-1;
	assert(isPrime(p) && p % 4 == 3);
	NDArray a = Ones((_, p), type<int>);
	-a;

	a[0] = 0;
	for(int i = 1; i < p; ++i) {
		a[(i * i) % p] = 1;
	}
	//a.print(std::cout);
	
	auto rep = Cycle(a, a.shape());
	NDArray A = Ones((_, n,n), type<int>);

	auto Q = Slice(A, A.shape(), (_, P{1,n}, P{1, n}));
	for(int i = 0; i < p; ++i) {
		for(int j = 0; j < p; ++j) {
			Q[i][j] = rep[j - i];
		}
	}
	Q -= Identity<int>(p);
	return A;
}

inline Ones<int, int, int> hadamardSylvester(int n) {
	if(n == 1) return Ones((_, 1, 1), type<int>);

	assert(isPowerOfTwo(n) && "power of two needed");
	
	using P = std::pair<int, int>;
	auto A = Ones((_, n, n), type<int>);
	
	auto B = hadamardSylvester(n / 2);
	//auto _B = Cycle(B, B.shape());
	//A.apply2([](int, int y) { return y; }, _B);
	//assign(A, A.shape(), _B);
	A.assign(Cycle(B, B.shape()));
	-Slice(A, A.shape(), (_, P{n / 2, n}, P{n / 2, n}));

	return A;
}

template<class T>
inline auto matrixToCode(T& A) {
	A.apply([](int x) {return x > 0;});
}


