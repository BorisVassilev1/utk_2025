#pragma once

#include <stdexcept>
#include "error.hpp"
#include "gauss.hpp"
#include "golay.hpp"
#include "ndarray.hpp"

class LinearCode {
	mutable bool d_computed = false;
	mutable int	 d;

	mutable bool r_computed = false;
	mutable int	 r = 0;

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

	int blockLength() { return std::get<0>(generator.shape()); }
	int length() { return std::get<1>(generator.shape()); }

	template <class Arr>
	auto encode(Arr &&c) {
		auto res = matmul_fancy(c, generator, type<int>);
		res.apply(mod2);
		return res;
	}

	bool isSelfOrthogonal() { return ::isSelfOrthogonal(generator); }
	int	 getDistance() {
		 if (!d_computed) {
			 d			= findDistance(generator);
			 d_computed = true;
		 }
		 return d;
	}

	int getCoverageRadius() {
		if (r_computed) return r;
		else {
			std::vector<NDArray<int, int>> sindromes;

			auto &&e = ErrorVectors(length(), blockLength());
			r = 0;
			for (auto it = e.begin(); it != e.end(); ++it) {
				NDArray<int, int, int> s = matmul_fancy(check, *it, type<int>);
				s.apply(mod2);

				NDArray<int, int> sind = NDArray((_, blockLength()), type<int>);
				for (std::size_t i = 0; i < sind.size(); ++i) {
					sind[i] = s[i][0];
				}

				bool found = false;
				for (auto &&entry : sindromes) {
					if (entry.operator==(sind)) {
						found = true;
						break;
					}
				}

				if (!found) {
					sindromes.emplace_back(NDArray((_, blockLength()), type<int>));
					sindromes.back() = sind;
					r				 = std::max(r, it.one_cnt);
				}
			}
			return r;
		}
	}
};
