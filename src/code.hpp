#pragma once

#include <stdexcept>
#include "gauss.hpp"
#include "ndarray.hpp"

class LinearCode {
   public:
	NDArray<int, int, int> generator;
	NDArray<int, int, int> check;

	template <class T>
		requires(T::Elem)
	LinearCode(T &&generator) : generator(std::forward<T>(generator)), check(orthogonal(generator)) {}

	LinearCode(std::istream &is) : generator((_, 1, 1), type<int>), check((_, 1, 1), type<int>) {
		std::string type;
		is >> type;
		if (type == "generator") {
			generator ^= NDArray((_, 1, 1), ::type<int>, is);
			check ^= orthogonal(generator);
		} else if (type == "check") {
			check ^= NDArray((_, 1, 1), ::type<int>, is);
			generator ^= orthogonal(check);
		} else throw std::runtime_error("invalid type of input for code");
	}

	void serializeGenerator(std::ostream &os) {
		os << "generator\n";
		generator.serialize(os);
	}
	void serializeCheck(std::ostream &os) {
		os << "check\n";
		check.serialize(os);
	}
};
