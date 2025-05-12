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
	code->generator.print(std::cerr);

	SindromeDecoder decoder(*code);
	
	int cnt = 0;
	auto arr = Zeros((_, code->length()), type<int>);
	char c = '\0';
	while((std::cin.get(c))) {
		if(c == '0' || c == '1') {
			arr[cnt] = c - '0'; 
			++cnt;

			if(cnt == code->length()) {
				std::cerr << "received: " << std::endl;
				for(int x : arr) std::cerr << x;

				std::cerr << std::endl;
				auto res = decoder.decode(arr);
				if(res.size() == 0) {
					std::cerr << "error" << std::endl;
					cnt = 0;
					continue;
				}

				std::cerr << "decoded: " << std::endl;
				for(int x : res) {
					std::cerr << x;
				}
				std::cerr << std::endl;
				cnt = 0;
			}
		}
	}
}
