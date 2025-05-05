#pragma once

inline bool isPrime(int n) {
	if (n < 2) return false;
	for (int i = 2; i * i <= n; ++i) {
		if (n % i == 0) return false;
	}
	return true;
}

inline bool isPowerOfTwo(int n) {
	return n > 0 && (n & (n - 1)) == 0;
}

