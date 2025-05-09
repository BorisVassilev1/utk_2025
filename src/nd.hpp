#pragma once

#include <functional>
#include <iomanip>
#include <iostream>
#include <ostream>
#include <tuple>

#include <utils.hpp>

template<class ...Args>
struct t : std::tuple<Args...> {
	t(Args &&... args) : std::tuple<Args...>(std::forward<Args>(args)...) {}
};

template <class... Args1, class... Args2>
std::tuple<Args1..., Args2...> operator,(std::tuple<Args1...> t1, std::tuple<Args2...> t2) {
	return std::tuple_cat(t1, t2);
}
template <class T, class... Args1>
std::tuple<Args1..., T> operator,(std::tuple<Args1...> t1, T i) {
	return std::tuple_cat(t1, std::make_tuple(i));
}

template <class T, class... Args1>
std::tuple<T, Args1...> operator,(T i, std::tuple<Args1...> t1) {
	return std::tuple_cat(std::make_tuple(i), t1);
}

inline std::tuple<int> operator""_t(unsigned long long int i) { return std::make_tuple((int)i); }

constexpr std::tuple<> _ = std::make_tuple();

template <typename Tuple, std::size_t... Is>
auto pop_front_impl(const Tuple &tuple, std::index_sequence<Is...>) {
	return std::make_tuple(std::get<1 + Is>(tuple)...);
}

template <typename Tuple>
auto pop_front(const Tuple &tuple) {
	return pop_front_impl(tuple, std::make_index_sequence<std::tuple_size<Tuple>::value - 1>());
}

template <std::size_t... n>
auto tuple_zeros_imp(std::index_sequence<n...>) {
	return std::make_tuple((n * 0.)...);
}
template <std::size_t N>
auto tuple_zeros() {
	return tuple_zeros_imp(std::make_index_sequence<N>());
}

template <std::size_t... S>
struct seq {};

template <std::size_t N, std::size_t... S>
struct gens : gens<N - 1, N - 1, S...> {};

template <std::size_t... S>
struct gens<0, S...> {
	typedef seq<S...> type;
};

template <template <typename...> class Tup1, template <typename...> class Tup2, typename... A, typename... B,
		  std::size_t... S>
auto tuple_zip_helper(Tup1<A...> t1, Tup2<B...> t2, seq<S...>)
	-> decltype(std::make_tuple(std::make_pair(std::get<S>(t1), std::get<S>(t2))...)) {
	return std::make_tuple(std::make_pair(std::get<S>(t1), std::get<S>(t2))...);
}

template <template <typename...> class Tup1, template <typename...> class Tup2, typename... A, typename... B>
auto tuple_zip(Tup1<A...> t1, Tup2<B...> t2)
	-> decltype(tuple_zip_helper(t1, t2, typename gens<sizeof...(A)>::type())) {
	static_assert(sizeof...(A) == sizeof...(B), "The tuple sizes must be the same");
	return tuple_zip_helper(t1, t2, typename gens<sizeof...(A)>::type());
}

template <class A, class B>
std::ostream &operator<<(std::ostream &out, const std::pair<A, B> &t) {
	return out << "(" << t.first << ", " << t.second << ")";
}

template <class... Args>
std::ostream &operator<<(std::ostream &out, const std::tuple<Args...> &t) {
	out << "(";
	[&]<std::size_t... p>(std::index_sequence<p...>) {
		((out << (p ? ", " : "") << std::get<p>(t)), ...);
	}(std::make_index_sequence<std::tuple_size_v<std::tuple<Args...>>>{});
	return out << ")";
}

template <typename T>
struct type_t {};
template <typename T>
inline type_t<T> type{};

template <class Container, class... Args>
class ND {
   public:
   protected:
	std::tuple<Args...> dimensions;

   private:
	Container		&This() { return *static_cast<Container *>(this); }
	const Container &This() const { return *static_cast<const Container *>(this); }

   public:
	ND(std::tuple<Args...> dim) : dimensions(dim) {
		static_assert((std::is_integral_v<Args> && ...), "ND constructor requires integral dimensions");
	}

	auto &print(std::ostream &out, int space = 2) { return This().print(out, space); }

	std::size_t			size() const { return size(dimensions); }
	std::tuple<Args...> shape() const { return dimensions; }

	template <class... Args2>
	static std::size_t size(std::tuple<Args2...> dim) {
		std::size_t size = 1;
		[&]<std::size_t... p>(std::index_sequence<p...>) {
			((size *= std::get<p>(dim)), ...);
		}(std::make_index_sequence<sizeof...(Args2)>{});
		return size;
	}

	void printInt(std::ostream &out, int space = 2)
		requires(sizeof...(Args) == 0)
	{
		out << std::setw(space) << This()[];
	}

	void printInt(std::ostream &out, int space = 2)
		requires(sizeof...(Args) > 0)
	{
		out << "[";
		std::size_t N = sizeof...(Args);
		bool		b = N == 2 && get<0>(dimensions) > 1;
		if (b) out << std::endl;
		for (int i = 0; i < get<0>(dimensions); ++i) {
			if (i) out << ",";
			if (i && b) out << std::endl;
			(*this)[i].printInt(out, space);
		}
		if (b) out << std::endl;
		out << "]";
	}

	template <class T>
	auto &operator=(T &&other)
		requires(sizeof...(Args) > 0)
	{
		if (This().shape() != other.shape()) { throw std::runtime_error("ND dimensions do not match"); }
		for (int i = 0; i < get<0>(shape()); ++i) {
			This()[i] = other[i];
		}
		return This();
	}

	template <class T>
	auto &operator=(const T &other)
		requires(sizeof...(Args) == 0 && std::is_fundamental_v<T>)
	{
		This()[] = other;
		return This();
	}

	template <class T>
	Container &operator=(T &&other)
		requires(sizeof...(Args) == 0 && !std::is_fundamental_v<T>)
	{
		This()[] = other[];
		return This();
	}

	template <class T>
	auto &assign(T &&other) {
		if (std::tuple_size_v<decltype(This().shape())> != std::tuple_size_v<decltype(other.shape())>) {
			throw std::runtime_error("Dimensions count does not match!");
		}
		for (int i = 0; i < get<0>(shape()); ++i) {
			This()[i].assign(other[i]);
		}
		return This();
	}
	template <class T>
	auto &assign(const T &other)
		requires(sizeof...(Args) == 0 && std::is_fundamental_v<T>)
	{
		This()[] = other;
		return This();
	}

	template <class T>
	Container &assign(T &&other)
		requires(sizeof...(Args) == 0 && !std::is_fundamental_v<T>)
	{
		This()[] = other[];
		return This();
	}

	template <class T>
	ND<Container, Args...> &apply(T &&f)
		requires(sizeof...(Args) > 0)
	{
		for (int i = 0; i < get<0>(shape()); ++i) {
			This()[i].apply(f);
		}
		return *this;
	}
	template <class T>
	ND<Container, Args...> &apply(T &&f)
		requires(sizeof...(Args) == 0)
	{
		This()[] = f(This()[]);
		return *this;
	}

	template <class T, class Arr>
	auto apply2(T &&f, Arr &&arr)
		requires(sizeof...(Args) > 0)
	{
		static_assert(std::tuple_size<decltype(arr.shape())>() == sizeof...(Args));
		for (int i = 0; i < get<0>(shape()); ++i) {
			This()[i].apply2(std::forward<T>(f), arr[i]);
		}
		return *this;
	}

	template <class T, class Arr>
	auto apply2(T &&f, Arr &&arr)
		requires(sizeof...(Args) == 0)
	{
		This()[] = f(This()[], arr[]);
		return *this;
	}

	template <class RHS>
	auto operator+=(RHS &&arr) {
		return apply2(std::plus(), arr);
	}
	template <class RHS>
	auto operator-=(RHS &&arr) {
		return apply2(std::minus(), arr);
	}

	template <class RHS>
	auto operator*=(RHS &&arr) {
		return apply2(std::multiplies(), arr);
	}

	template <class RHS>
	auto multScalar(RHS &&v) {
		return apply([&v]<class T>(T &x) { return x * v; });
	}

	auto operator-() {
		return apply(std::negate());
	}

	template <class T>
	ND<Container, Args...> &foreach (T &&f) {
		for (std::size_t i = 0; i < size(); ++i) {
			f(This().getLinear(i));
		}
		return *this;
	}

	void serialize(std::ostream &out) {
		[&]<auto... p>(std::index_sequence<p...>) {
			((out << std::get<p>(shape()) << " "), ...);
		}(std::make_index_sequence<sizeof...(Args)>{});
		out << "%%%";
		serializeData(out);
	}

	void serializeData(std::ostream &out)
		requires(sizeof...(Args) > 0)
	{
		for (int i = 0; i < std::get<0>(shape()); ++i) {
			This()[i].serializeData(out);
		}
	}

	void serializeData(std::ostream &out)
		requires(sizeof...(Args) == 0)
	{
		out << This()[] << " ";
	}

	template <class T>
	bool operator==(const T &other)
		requires(sizeof...(Args) == 0 && std::is_fundamental_v<T>)
	{
		return This()[] == other;
	}
	template <class T>
	bool operator==(T &&other)
		requires(sizeof...(Args) == 0 && !std::is_fundamental_v<T>)
	{
		return This()[] == other[];
	}

	template <class T>
	bool operator==(T &&other)
		requires(sizeof...(Args) > 0)
	{
		if (shape() != other.shape()) { return false; }
		for (int i = 0; i < std::get<0>(shape()); ++i) {
			if (!(This()[i].operator==(other[i]))) { return false; }
		}
		return true;
	}

	auto operator[](int index)
		requires(sizeof...(Args) > 0)
	{
		return This()[index];
	}
	auto operator[](Args... indices)
		requires(sizeof...(indices) == sizeof...(Args) && sizeof...(indices) != 1)
	{
		return This()[indices...];
	}
	auto slice(std::size_t i, std::size_t j) { return This().slice(i, j); }
};
