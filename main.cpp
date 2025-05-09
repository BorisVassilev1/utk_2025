#include <cmath>
#include <ios>
#include <iostream>
#include <fstream>

#include <ndarray.hpp>
#include <primitives.hpp>
#include <slice.hpp>
#include <tuple>
#include <utils.hpp>
#include <hadamard.hpp>
#include <gauss.hpp>
#include <golay.hpp>

using P = std::pair<int, int>;

int main() {
	{
		std::cout << "------------ Hadamard Matrices ------------------" << std::endl;
		auto A = hadamardPaley(12);
		matrixToCode(A);
		A.print(std::cout, 1);

		auto B = hadamardSylvester(16);
		matrixToCode(B);
		B.print(std::cout, 1);

		{
			auto M = NDArray((_, 8, 16), type<int>);
			auto Reg = Slice(B, B.shape(), (_, P{8, 16}, P{0, 16}));
			assign(M, M.shape(), Reg);
			M.print(std::cout);
			gaussSolve(M);
			std::cout << codeInfo(M) << std::endl;
		}

		auto C = hadamardPaley(20);
		matrixToCode(C);
		C.print(std::cout, 1);

		auto D = hadamardPaley(24);
		matrixToCode(D);
		D.print(std::cout, 1);

		{
			std::ofstream out("hadamard_12.txt");
			A.serialize(out);
		}

		{
			std::ifstream in("hadamard_12.txt");
			auto		  D = NDArray((_, 0, 0), type<int>, in);
			assert(D.shape() == A.shape());
			assert(D.operator==(A));
		}

		{
			std::ofstream out("hadamard_16.txt");
			B.serialize(out);
		}
		{
			std::ofstream out("hadamard_24.txt");
			C.serialize(out);
		}
	}

	{
		std::cout << "------------- Gauss Elimination -----------------" << std::endl;
		int		n = 3;
		int		m = 4;
		NDArray G((_, n, m), type<int>);
		G[0][0] = 1;
		G[0][1] = 1;
		G[0][2] = 0;
		G[0][3] = 0;
		G[1][0] = 1;
		G[1][1] = 0;
		G[1][2] = 1;
		G[1][3] = 0;
		G[2][0] = 1;
		G[2][1] = 0;
		G[2][2] = 0;
		G[2][3] = 1;
		G.print(std::cout);

		auto H = orthogonal(G);
		H.print(std::cout);
	}

	{
		std::cout << "--------------- Golay Codes -------------------" << std::endl;
		NDArray G = Golay24();
		G.print(std::cout);

		std::cout << std::boolalpha << "is G orthogonal to G? :" << isSelfOrthogonal(G) << std::endl;

		auto H = orthogonal(G);
		gaussSolve(H);
		std::cout << "H =?= G : " << (H.operator==(G)) << std::endl;

		{
			auto [n, m, k] = codeInfo(G);
			std::cout << "G: [" << n << ", " << m << ", " << k << "]" << std::endl;
		}

		{
			auto [n, m, k] = codeInfo(Slice(G, G.shape(), (_, P{0, 12}, P{0, 23})));
			std::cout << "G*: [" << n << ", " << m << ", " << k << "]" << std::endl;
		}
	}
	{
		std::ifstream in("P.txt");
		NDArray		  K = NDArray((_, 1, 1), type<int>, in);
		K.print(std::cout);

		gaussSolve(K);
	
		K.print(std::cout);
		CODE_INFO(K);
		
		auto K_orthogonal = orthogonal(K);
		K_orthogonal.print(std::cout);
		CODE_INFO(K_orthogonal);

	}
}
