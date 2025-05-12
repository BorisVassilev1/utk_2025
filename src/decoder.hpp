#pragma once

#include "code.hpp"
#include "nd.hpp"
#include "ndarray.hpp"
#include "primitives.hpp"

class SindromeDecoder {
	struct TableEntry {
		NDArray<int, int> e;
		NDArray<int, int> sindrome;

		TableEntry(NDArray<int, int> e, NDArray<int, int> sindrome) : e(e), sindrome(sindrome) {}
	};

	std::vector<TableEntry> sindromes;
	LinearCode			   &code;

   public:
	SindromeDecoder(LinearCode &code) : code(code) {
		int dist = code.getDistance();
		std::cerr << std::format("code distance is: [{}, {}, {}] \n r(C) = {}", code.length(), code.blockLength(), dist,
								 code.getCoverageRadius())
				  << std::endl;

		auto s0 = matmul_fancy(code.check, code.generator[0], type<int>);
		s0.apply(mod2);

		int t = (dist - 1) / 2;

		for (auto &&e : ErrorVectors(code.length(), t)) {
			// e.print(std::cout);
			NDArray<int, int, int> s = matmul_fancy(code.check, e, type<int>);
			s.apply(mod2);

			sindromes.emplace_back(e, NDArray((_, code.blockLength()), type<int>));

			NDArray<int, int> &sind = sindromes.back().sindrome;
			for (std::size_t i = 0; i < sind.size(); ++i) {
				sind[i] = s[i][0];
			}
		}
	}

	auto decode(NDArray<int, int> &codeword) {
		auto s = matmul_fancy(code.check, codeword, type<int>);
		s.apply(mod2);
		NDArray sind = NDArray((_, code.blockLength()), type<int>);
		for (std::size_t i = 0; i < sind.size(); ++i) {
			sind[i] = s[i][0];
		}

		for (auto &&entry : sindromes) {
			if (entry.sindrome.operator==(sind)) {
				codeword += entry.e;
				codeword.apply(mod2);
				
				auto y = solve(code.generator, codeword);

				return y;
			}
		}
		std::cerr << "failed decoding" << std::endl;
		std::cerr << std::endl;
		return NDArray((_, 0), type<int>);
	}
};
