#pragma once
#include "ndarray.hpp"
#include "primitives.hpp"

class ErrorVectors {
   public:
	class Iterator {
		NDArray<int, int> vec;

	   public:
		int one_cnt;
		Iterator(int size) : vec(Zeros((_, size), type<int>)), one_cnt(0) {}

		bool operator!=(const Iterator &other) const { return one_cnt != other.one_cnt; }
		bool operator==(const Iterator &other) const { return one_cnt == other.one_cnt; }

		bool operator!=(const int &other) const { return one_cnt != other; }
		void operator++() {
			bool b = std::prev_permutation(vec.begin(), vec.end());
			if (!b) { vec[one_cnt++] = 1; }
		}

		auto &operator*() { return vec; }
	};

	int size;
	int max_cnt;

   public:
	ErrorVectors(int size, int max_cnt) : size(size), max_cnt(max_cnt) {
		if (max_cnt > size) { throw std::runtime_error("max_cnt > size"); }
	}

	auto begin() const { return Iterator(size); }
	auto end() const { return max_cnt + 1; }
};
