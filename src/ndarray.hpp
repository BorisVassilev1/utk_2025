#pragma once
#include <iostream>
#include <ostream>
#include <tuple>
#include <cassert>

#include "autoref.hpp"
#include <nd.hpp>

template <class T, class... Args>
class NDArray : public ND<NDArray<T, Args...>, Args...> {
	using Parent = ND<NDArray<T, Args...>, Args...>;

	NDArray(std::tuple<Args...> dim, T *data, std::size_t offset = 0);

   public:
	using Elem = T;
	NDArray(std::tuple<Args...> dim, type_t<T> t = type<T>);
	NDArray(std::tuple<Args...> dim, type_t<T> t, std::istream &in);
	NDArray(NDArray &other);
	NDArray(NDArray &&other) = default;

	NDArray &operator=(NDArray &other) { return Parent::operator=(other); }
	NDArray &operator=(NDArray &&other) { return Parent::operator=(other); }
	NDArray &operator=(const T &other) { return Parent::operator=(other); }
	template <class U>
	NDArray &operator=(U &&other) {return Parent::template operator=<U>(std::forward<U>(other));}

	T &operator[](Args... indices)
		requires(sizeof...(indices) == sizeof...(Args) && sizeof...(indices) != 1)
	{
		static_assert((std::is_integral_v<Args> && ...), "ND operator[] requires integral dimensions");

		std::size_t			  index		 = offset;
		std::size_t			  multiplier = 1;
		constexpr std::size_t N			 = sizeof...(Args) - 1;
		std::tuple<Args...>	  ind		 = std::tuple<Args...>(indices...);

		[&]<std::size_t... p>(std::index_sequence<p...>) {
			((index += std::get<N - p>(ind) * multiplier, multiplier *= std::get<N - p>(this->dimensions)), ...);
		}(std::make_index_sequence<sizeof...(Args)>{});

		return data[index];
	}

	void print(std::ostream &out, int space = 2) {
		std::cout << "NDArray" << this->dimensions << std::endl;
		this->printInt(out, space);
		std::cout << std::endl;
	}

	operator T &()
		requires(sizeof...(Args) == 0)
	{
		return data[offset];
	}

	auto operator[](int index)
		requires(sizeof...(Args) > 0)
	{
		std::tuple	t	   = pop_front(this->dimensions);
		std::size_t offset = this->offset + index * this->size(t);
		return ::NDArray(t, (T *)data, offset);
	}

	template <class U, class... Args2>
	friend class NDArray;

	SmartRefArr<T> data;

   protected:
	std::size_t offset;
};

template <class T, class... Args>
NDArray(std::tuple<Args...>, type_t<T>) -> NDArray<T, Args...>;

template <class T, class... Args>
NDArray(std::tuple<Args...>, T *, std::size_t, std::size_t) -> NDArray<T, Args...>;

template <class T, class... Args>
NDArray(std::tuple<Args...>, type_t<T>, std::istream &) -> NDArray<T, Args...>;

// --------------------------------------------------------------------------------------------------------------------

template <class T, class... Args>
NDArray<T, Args...>::NDArray(std::tuple<Args...> dim, T *data, std::size_t offset)
	: Parent(dim), data(*data), offset(offset) {
	static_assert((std::is_integral_v<Args> && ...), "ND constructor requires integral dimensions");
};

template <class T, class... Args>
NDArray<T, Args...>::NDArray(std::tuple<Args...> dim, type_t<T>)
	: Parent(dim), data(new T[Parent::size()]), offset(0) {
	static_assert((std::is_integral_v<Args> && ...), "ND constructor requires integral dimensions");
}

template <class T, class... Args>
NDArray<T, Args...>::NDArray(NDArray &other)
	: Parent(other.dimensions), data(&other.data + other.offset, other.size()), offset(other.offset) {
	assert(data.isRef == false);
	static_assert((std::is_integral_v<Args> && ...), "ND copy constructor requires integral dimensions");
}

template <class T, class... Args>
NDArray<T, Args...>::NDArray(std::tuple<Args...> dim, type_t<T>, std::istream &in)
	: Parent(dim), data(nullptr), offset(0) {
	[&]<auto... p>(std::index_sequence<p...>) {
		((in >> std::get<p>(this->dimensions)), ...);
	}(std::make_index_sequence<sizeof...(Args)>{});
	while (in.peek() == '%' || std::isspace(in.peek()))
		in.ignore();
	data = new T[Parent::size()];
	for (std::size_t i = 0; i < this->size(); ++i) {
		in >> data[i];
	}
}
