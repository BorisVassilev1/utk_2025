#pragma once

#include "code.hpp"
#include "primitives.hpp"

class SindromeDecoder {
	struct TableEntry {
		NDArray<int, int> e;
		NDArray<int, int> sindrome;
	};

	std::vector<TableEntry> sindromes;
public:
	SindromeDecoder(LinearCode &code)
	{
		int dist = code.getDistance();
		std::cout << "code distance is: " <<dist << std::endl;

		auto s0 = matmul_fancy(code.check, code.generator[0], type<int>);
		// TODO: finish construction of table
		s0.apply(mod2);
		s0.print(std::cout);
	}
};
