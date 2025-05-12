#include <iostream>
#include <fstream>
#include <memory>
#include "code.hpp"

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
	//code->generator.print(std::clog);
	
	int cnt = 0;
	auto arr = Zeros((_, code->blockLength()), type<int>);
	char c;
	while((std::cin.get(c))) {
		if(c == '0' || c == '1') {
			arr[cnt] = c - '0'; 
			++cnt;

			if(cnt == code->blockLength()) {
				auto res = code->encode(arr);
				for(int x : res[0]) std::cout << x;
				std::cout << std::endl;
				
				std::clog << "sent: " << std::endl;
				for(int x : res[0]) std::clog << x;
				std::clog << std::endl;

				cnt = 0;
			}
		}
	}
}
