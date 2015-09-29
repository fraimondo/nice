#include <blocktrie.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <symb_transf.h>



int main(char argc, char* argv[]) {
	nsymbols = 0;
	int kernel = 3;

	n_total_symbols = 1;
    for (int c = 1; c <= kernel; c++) {
    	n_total_symbols = n_total_symbols * c;
    }


	unsigned char * symbolstr = malloc(kernel+1 * sizeof(char));
	for (int i = 0; i < kernel; i++) {
		symbolstr[i] = i;
	}
	trie_t * symbols = trie_create(kernel);
	
	definePermutations(symbolstr, kernel, symbols);

	trie_dinfo(symbols);
	
	free(symbolstr);


	printf("Value for 123 = %d\n", trie_defined(symbols, "\0\1\2", 0, 3));
	printf("Value for 132 = %d\n", trie_defined(symbols, "\0\2\1", 0, 3));
	printf("Value for 213 = %d\n", trie_defined(symbols, "\1\0\2", 0, 3));
	printf("Value for 312 = %d\n", trie_defined(symbols, "\2\0\1", 0, 3));
	printf("Value for 231 = %d\n", trie_defined(symbols, "\1\2\0", 0, 3));
	printf("Value for 321 = %d\n", trie_defined(symbols, "\2\1\0", 0, 3));
}