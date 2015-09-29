#ifndef __SMI_H__
#define __SMI_H__

#include <types.h>

/*
 * Compute SMI and wSMI on symbolic transformed data.
 *
 * Uses OpenMP to parallelize across trials.
 *
 * data: input data, must be symbolic with values in the range [0, nsymbols-1]
 *		 nchannels by nsamples by ntrials C ordered matrix.
 *
 * count: the probability of each symbol.
 *        nchannels by nsymbols by ntrials C ordered matrix.
 *
 * wts: the weight matrix to use in the wSMI computation
 *		nsymbols by nsymbols C ordered matrix.
 *
 * mi: Symbolic Mutual Information result.
 *	   nchannels by nchannels by ntrials upper diagonal C ordered matrix.
 *
 * wmi: Weighted Symbolic Mutual Information result.
 *	   nchannels by nchannels by ntrials upper diagonal C ordered matrix.
 *
 * nchannels: number of channels in the data.
 * nsamples: number of samples in the data.
 * ntrials: number of trials in the data.
 * nsymbols: number of symbols in the data.
 * nthreads: amount of threads to use in the computation.
 */
int smi(
		int * data,
		double * count,
		double * wts,
		double * mi,
		double * wmi,
		int nchannels,
		int nsamples,
		int ntrials,
		int nsymbols,
		int nthreads
	);

#endif
