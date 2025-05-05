#pragma once
#include <nd.hpp>
#include <autoref.hpp>
#include <utility>
#include <utils.hpp>
#include "ndarray.hpp"

template <class Container, class... Args>
class Slice : public ND<Slice<Container, Args...>, Args...> {
	AutoRef<Container> data;
	using Parent = ND<Slice<Container, Args...>, Args...>;

	std::tuple<Args...> offset;
	std::tuple<Args...> cap;

   public:
	template <class U, class... Constraints>
	Slice(U &&array, std::tuple<Args...>, std::tuple<Constraints...> c)
		: Parent(sub_v(snd_v(c), fst_v(c))), data(std::forward<U>(array)), offset(fst_v(c)), cap(snd_v(c)) {
		static_assert((std::is_integral_v<Args> && ...), "ND constructor requires integral dimensions");
	}

	auto operator[](std::size_t index)
		requires(sizeof...(Args) > 0)
	{
		std::size_t i = index + get<0>(this->offset);
		auto res = (*data)[i];
		auto s = ::Slice(std::move(res), res.shape(), pop_front(tuple_zip(offset, cap)));
		//std::cout << "Slice " << type_name<decltype(s)>() << " " << s.shape() << std::endl;
		return s;
	}

	auto &operator[](Args... indices) {
		auto ind = add_v(std::forward_as_tuple(indices...), this->offset);
		return [&]<size_t...p>(std::index_sequence<p...>) -> auto& {
			return (*data)[(std::get<p>(ind))...];
		}(std::make_index_sequence<sizeof...(Args)>());
	}

	template <class... P>
	static std::tuple<Args...> fst_v(std::tuple<P...> t) {
		return [&]<auto... p>(std::index_sequence<p...>) {
			return std::make_tuple((std::get<0>(std::get<p>(t))) ...);
		}(std::make_index_sequence<sizeof...(Args)>{});
	}

	template <class... P>
	static std::tuple<Args...> snd_v(std::tuple<P...> t) {
		return [&]<auto... p>(std::index_sequence<p...>) {
			return std::make_tuple((std::get<1>(std::get<p>(t))) ...);
		}(std::make_index_sequence<sizeof...(Args)>{});
	}

	static std::tuple<Args...> add_v(std::tuple<Args...> a, std::tuple<Args...> b) {
		[&]<auto... p>(std::index_sequence<p...>) {
			((std::get<p>(a) += std::get<p>(b)), ...);
		}(std::make_index_sequence<sizeof...(Args)>{});
		return a;
	}
	static std::tuple<Args...> sub_v(std::tuple<Args...> a, std::tuple<Args...> b) {
		[&]<auto... p>(std::index_sequence<p...>) {
			((std::get<p>(a) -= std::get<p>(b)), ...);
		}(std::make_index_sequence<sizeof...(Args)>{});
		return a;
	}

	void print(std::ostream &out, int space = 2) {
		std::cout << "Slice" << this->dimensions << std::endl;
		this->printInt(out, space);
		std::cout << std::endl;
	}

	using Parent::operator=;
};


template <class U, class... Args, class ...Constraints>
Slice(U &&, std::tuple<Args...>, std::tuple<Constraints...>) -> Slice<U, Args...>;

