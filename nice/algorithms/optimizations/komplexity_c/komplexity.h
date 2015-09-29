#ifndef __KOMPLEXITY_HH__
#define __KOMPLEXITY_HH__
/*
 * Calculate the complexity for all the channels and trials.
 *
 * data: array of double with size nchannels by nsamples by ntrials
 *       it should be stored in COLUMN MAJOR ORDER. So every sample in
 *       each channel is stored contigously.
 *
 * nchannels: number of channels in the data.
 * nsamples: number of samples in the data.
 * ntrials: number of trials in the data.
 *
 * results: array used to store the results.
 *          Size: ntrials * nchannels * sizeof(double)
 *
 * nbins: number of bins used to compute the symbolic transformation.
 */
int do_process_all_trials(
	double * data,
	long nsamples,
	long nchannels,
	long ntrials,
	double * results,
	long nbins,
	int nthreads
	);

#endif