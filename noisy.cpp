#include <iostream>
#include <memory>
#include "code.hpp"
#include "decoder.hpp"
#include <fstream>

int main(int argc, char** argv) {


	std::unique_ptr<LinearCode> code;
	if(argc == 2) {
		code = std::make_unique<LinearCode>(std::cin);
	}
	else if(argc == 3) {
		std::ifstream in(argv[2]);
		code = std::make_unique<LinearCode>(in);
	}
	else {
		std::cerr << "1 or 2 arguments needed" << std::endl;
		exit(1);
	}
	int errCnt = std::stoi(argv[1]);

	srand(time(0));

	int cnt = 0;
	auto arr = NDArray((_, code->length()), type<int>);
	char c = '\0';
	while((std::cin.get(c))) {
		if(c == '0' || c == '1') {
			arr[cnt] = c - '0'; 
			++cnt;

			if(cnt == code->length()) {
				for(int i = 0; i < errCnt; ++i) {
					int pos = rand() % code->length();
					arr[pos] = 1 - arr[pos];
				}
				
				for(int x : arr) {
					std::cout << x;
				}
				cnt = 0;
			}
		}
	}
}
