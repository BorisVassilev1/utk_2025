#pragma once
#include <ndarray.hpp>
#include <tuple>
#include <type_traits>
#include "autoref.hpp"

template <class T, class... Args>
	requires std::is_fundamental<T>::value
class Zeros : public NDArray<int, Args...> {
   public:
	Zeros(std::tuple<Args...> dim, type_t<T> t = type<int>) : NDArray<int, Args...>(dim) {
		memset(&this->data, 0, sizeof(T) * this->size());
	}
	using NDArray<int, Args...>::operator=;
};

template <class T, class... Args>
	requires std::is_fundamental<T>::value
class Ones : public NDArray<T, Args...> {
   public:
	Ones(std::tuple<Args...> dim, type_t<T> t = type<int>) : NDArray<T, Args...>(dim) {
		for (std::size_t i = 0; i < this->size(); ++i) {
			this->data[i] = 1;
		}
	}
	//using NDArray<T, Args...>::operator[];
	using NDArray<T, Args...>::operator=;
};

template <class T, class U, class... Args>
class Repeat : public NDArray<T, int, Args...> {
	using Parent = NDArray<T, int, Args...>;

   public:
	using Parent::operator=;
	Repeat(const ND<U, Args...> &array, std::size_t dim, type_t<T> t) : Parent((dim, array.shape()), type<T>) {
		for (std::size_t i = 0; i < dim; ++i) {
			(*this)[i].template operator= <decltype(array)>(array);
		}
	}
};

inline int mod(int k, int n) { return ((k %= n) < 0) ? k + n : k; }

template <class Container, class... Args>
class Cycle : public ND<Cycle<Container, Args...>, Args...> {
	AutoRef<Container> data;
	using Parent = ND<Cycle<Container, Args...>, Args...>;

   public:
	template <class U>
	Cycle(U &&array, std::tuple<Args...>) : Parent(array.shape()), data(std::forward<U>(array)) {
		static_assert((std::is_integral_v<Args> && ...), "ND constructor requires integral dimensions");
	}

	auto operator[](std::size_t index)
		requires(sizeof...(Args) > 0)
	{
		std::size_t i = ::mod(index, get<0>(this->shape()));
		assert(i <= get<0>(this->shape()));
		return ::Cycle(std::move((*data)[i]), pop_front(this->shape()));
	}
	auto &operator[](Args... indices) {
		auto ind = mod_v(std::forward_as_tuple(indices...), this->shape());
		return [&]<size_t... p>(std::index_sequence<p...>) -> auto & {
			return (*data)[(std::get<p>(ind))...];
		}(std::make_index_sequence<sizeof...(Args)>());
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

template <class T, class U>
class Vectorize {
   public:
	Vectorize(const std::function<T(U)> &f) { this->f = f; }

	template <class X>
	auto operator()(X &&x) {
		auto result = NDArray(x.shape(), type<T>);
		for (std::size_t i = 0; i < x.size(); ++i) {
			result.getLinear(i) = f(x.getLinear(i));
		}
		return result;
	}

   private:
	std::function<T(U)> f;
};

template <class T>
T negate(T x) {
	return -x;
}

template <class T>
auto negate_v = Vectorize<T, T>(negate<T>);

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
	for (std::size_t i = 0; i < std::get<0>(shape); ++i) {
		for (std::size_t j = 0; j < std::get<1>(shape); ++j) {
			u[i][j] = v[j][i];
		}
	}
}

template <class U, class V>
void assign(U &&u, std::tuple<int, int> shape, V &&v) {
	for (std::size_t i = 0; i < std::get<0>(shape); ++i) {
		for (std::size_t j = 0; j < std::get<1>(shape); ++j) {
			u[i][j] = v[i][j];
		}
	}
}
