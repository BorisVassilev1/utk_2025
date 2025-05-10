#include <iostream>
#include <memory>
#include "code.hpp"
#include "decoder.hpp"
#include <fstream>

int main(int argc, char** argv) {


	std::unique_ptr<LinearCode> code;
	if(argc == 1) {
		code = std::make_unique<LinearCode>(std::cin);
	}
	else if(argc == 2) {
		std::ifstream in(argv[1]);
		code = std::make_unique<LinearCode>(in);
	}
	else {
		std::cerr << "only one argument needed" << std::endl;
		exit(1);
	}
	code->generator.print(std::cout);

	SindromeDecoder decoder(*code);
	
	int cnt = 0;
	auto arr = NDArray((_, code->blockLength()), type<int>);
	char c = '\0';
	while((std::cin.get(c))) {
		if(c == '0' || c == '1') {
			arr[cnt] = c - '0'; 
			++cnt;

			if(cnt == code->blockLength()) {
				//auto res = code->encode(arr);
				//for(int i = 0; i < code->length(); ++i) {
				//	std::cout << res[0][i];
				//}
				//std::cout << std::endl;
				//cnt = 0;
			}
		}
	}
}
