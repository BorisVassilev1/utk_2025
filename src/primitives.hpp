#pragma once
#include <cmath>
#include <exception>
#include <ndarray.hpp>
#include <tuple>
#include <type_traits>
#include "autoref.hpp"

inline int mod2(int x) {
	return x & 1;
}

template <class T, class... Args>
	requires std::is_fundamental<T>::value
class Zeros : public NDArray<int, Args...> {
   public:
	Zeros(std::tuple<Args...> dim, type_t<T> = type<int>) : NDArray<int, Args...>(dim) {
		memset(&this->data, 0, sizeof(T) * this->size());
	}
	using NDArray<int, Args...>::operator=;

	using Elem = int;
};

template <class T, class... Args>
	requires std::is_fundamental<T>::value
class Ones : public NDArray<T, Args...> {
   public:
	Ones(std::tuple<Args...> dim, type_t<T> = type<int>) : NDArray<T, Args...>(dim) {
		for (std::size_t i = 0; i < this->size(); ++i) {
			this->data[i] = 1;
		}
	}
	using ND<NDArray<T, Args...>, Args...>::operator=;
	using Elem = T;
};

template <class T, class... Args>
class Repeat : public NDArray<T, int, Args...> {
	using Parent = NDArray<T, int, Args...>;

   public:
	using Parent::operator=;
	template<class U>
	Repeat(U&& array, std::size_t dim, std::tuple<Args...>, type_t<T>) : Parent((dim, array.shape()), type<T>) {
		for (std::size_t i = 0; i < dim; ++i) {
			(*this)[i].template operator= <decltype(array)>(array);
		}
	}
	using Elem = T;
};

template<class T, class U, class ...Args>
Repeat(U&& array, std::size_t dim, std::tuple<Args...>, type_t<T>) -> Repeat<T, Args...>;

inline int mod(int k, int n) { return ((k %= n) < 0) ? k + n : k; }

template <class Container, class... Args>
class Cycle : public ND<Cycle<Container, Args...>, Args...> {
	AutoRef<Container> data;
	using Parent = ND<Cycle<Container, Args...>, Args...>;

   public:
	using Elem = std::remove_reference_t<Container>::Elem;

	template <class U>
	Cycle(U &&array, std::tuple<Args...>) : Parent(array.shape()), data(std::forward<U>(array)) {
		static_assert((std::is_integral_v<Args> && ...), "ND constructor requires integral dimensions");
	}

	auto operator[](std::size_t index)
		requires(sizeof...(Args) > 0)
	{
		int i = ::mod(index, get<0>(this->shape()));
		assert(i <= get<0>(this->shape()));
		return ::Cycle(std::move((*data)[i]), pop_front(this->shape()));
	}
	auto &operator[](Args... indices)
		requires(sizeof...(indices) == sizeof...(Args) && sizeof...(indices) != 1)
	{
		auto ind = mod_v(std::forward_as_tuple(indices...), this->shape());
		return [&]<size_t... p>(std::index_sequence<p...>) -> auto & {
			return (*data)[(std::get<p>(ind))...];
		}(std::make_index_sequence<sizeof...(Args)>());
	}

	operator Elem&() {
		return data.get();
	}

	static std::tuple<Args...> mod_v(std::tuple<Args...> ind, std::tuple<Args...> dim) {
		[&]<auto... p>(std::index_sequence<p...>) {
			((std::get<p>(ind) = mod(std::get<p>(ind), std::get<p>(dim))), ...);
		}(std::make_index_sequence<sizeof...(Args)>{});
		return ind;
	}
};
template <class U, class... Args>
Cycle(U &&, std::tuple<Args...>) -> Cycle<U, Args...>;

template<class Container, class ...Args>
class Transpose : public ND<Transpose<Container, Args...>, Args...> {
	AutoRef<Container> data;
	using Parent = ND<Transpose<Container, Args...>, Args...>;

public:
	using Elem = std::remove_reference_t<Container>::Elem;

};


template <class T>
struct Eye : Zeros<T, int> {
   public:
	Eye(std::size_t n, std::size_t i) : Zeros<T, int>((_, n), type<T>) { (*this)[i] = 1; }
};

template <class T>
struct Identity : Zeros<T, int, int> {
   public:
	Identity(std::size_t n) : Zeros<T, int, int>((_, n, n), type<T>) {
		for (std::size_t i = 0; i < n; ++i)
			(*this)[i][i] = 1;
	}
};

template <class U, class V>
void assignTransposed(U &&u, std::tuple<int, int> shape, V &&v) {
	for (int i = 0; i < std::get<0>(shape); ++i) {
		for (int j = 0; j < std::get<1>(shape); ++j) {
			u[i][j] = v[j][i];
		}
	}
}

template <class U, class V>
void assign(U &&u, std::tuple<int, int> shape, V &&v) {
	for (int i = 0; i < std::get<0>(shape); ++i) {
		for (int j = 0; j < std::get<1>(shape); ++j) {
			u[i][j] = v[i][j];
		}
	}
}

template<class A, class B>
int dot(A && a, B && b) {
	int res = 0;
	for(int i = 0; i < std::get<0>(a.shape()); ++i) {
		res += a[i] * b[i];
	}
	return res;
}

/// [k] x [k, n] -> [n]
template<class A, class M>
auto vecMatMul(A && a, M && m) {
	auto [k1] = a.shape();
	auto [k2, n] = m.shape();
	assert(k1 == k2 && "dimensions must match");
	auto k = k1;
	auto res = Zeros((_, n), type<int>);

	auto exp1 = Zeros((_, n), type<int>);
	for(int i = 0; i < k; ++i) {
		exp1 = m[i];
		exp1.multScalar(a[i]);
		res += exp1;
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
template<class U, class V, class T>
auto matmul(U && u, V && v, type_t<T> = type<T>) {
	auto [n, m1] = u.shape();
	auto [m2, k] = v.shape();
	if(m1 != m2) {
		throw std::runtime_error(std::format("dimensions must match: [{}, {}] and [{}, {}]", n, m1, m2, k));
	}
	assert(m1 == m2 && "dimensions must match");

	auto vt = NDArray((_, k, m2), type<T>);
	assignTransposed(vt, vt.shape(), v);
	auto res = NDArray((_, n, k), type<T>);
	for(int i = 0; i < n; ++i) 
		for(int j = 0; j < k; ++j)
			res[i][j] = dot(u[i], vt[j]);

	return res;
}

template<class U, class V, class T>
auto matmul_fancy(U && u, V && v, type_t<T> = type<T>) {
	constexpr bool u1 = std::tuple_size_v<decltype(u.shape())> == 1;
	constexpr bool v1 = std::tuple_size_v<decltype(v.shape())> == 1;

	if constexpr (u1 && v1) {
		auto res = NDArray(_, type<int>);
		res[] = dot(u, v);
	} else if constexpr (u1) {
		return matmul(Repeat(u, 1, u.shape(), type<T>), v, type<T>); 
	} else if constexpr (v1) {
		int l = std::get<0>(v.shape());
		auto my_v = NDArray((_, l, 1), type<int>);
		for(int i = 0; i < l; ++i) {
			my_v[i][0] = v[i];
		}
		return matmul(u, my_v, type<T>); 
	} else {
		return matmul(u, v);
	}

}
