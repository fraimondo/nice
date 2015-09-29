#ifndef __SYMB_TRANSF_H__
#define __SYMB_TRANSF_H__

#include <types.h>
#include <blocktrie.h>

void definePermutations(unsigned char *str, int len, trie_t * symbols);

extern int nsymbols;
extern int n_total_symbols;

/*
 * Generate a symbolic transformation on the data .
 *
 * data: input data, must be filtered with a lowpass at sampling_freq/kernel/tau.
 *		 nchannels by nsamples by ntrials C ordered matrix.
 *
 * kernel: number of samples to transform into a single symbol.
 *
 * tau: number of samples between each sample to account in the transformation.
 *
 * signal_symb_transf: the symbolic transformation returned.
 *			  nchannels by n_symbol_samples by ntrials C ordered matrix.
 *			  n_symbol_samples = nsamples - tau * (kernel - 1);
 *
 * count: the probability of each symbol.
 *        nchannels by nsymbols by ntrials C ordered matrix.
 *        nsymbols = kernel!
 *
 * nchannels: number of channels in the data.
 * nsamples: number of samples in the data.
 * ntrials: number of trials in the data.
 */
int symb_transf(
		double * data,
		int kernel,
		int tau,
		int * signal_symb_transf,
		double * count,
		int nchannels,
		int nsamples,
		int ntrials
	) ;

#endif
