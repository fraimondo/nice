#ifndef __OMP_KOMPLEXITY_MAIN__
#define __OMP_KOMPLEXITY_MAIN__

#include <Python.h>


#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL jivaro_main_array_symbol

#ifndef __OMP_KOMPLEXITY_MAIN__
#define NO_IMPORT_ARRAY
#endif

#include <types.h>

#include <numpy/arrayobject.h>
#include <komplexity.h>
#include <helpers.h>

static PyObject *ompkError;


static PyObject * ompk_komplexity(PyObject *self, PyObject *args) {
	PyArrayObject *data;
	int nbins;
	int nthreads;
	if (!PyArg_ParseTuple(args, "Oii", &data, &nbins, &nthreads)) {
		PyErr_SetString(ompkError, "Invalid parameters.");
	}
	// printf ("Calling pe with %p, kernel %d and tau %d\n", data, kernel, tau);
	// PyArrayObject * signal_symb = NULL;
	// PyArrayObject * count = NULL;


	npy_intp * dims = PyArray_DIMS(data);

	int ntrials = dims[0];
	int nchannels = dims[1];
	int nsamples = dims[2];

	double * c_data = malloc(nchannels * nsamples * ntrials * sizeof(double));

	int trial;
	int channel;
	int sample;
	for (trial = 0; trial < ntrials; trial++) {
		for (channel = 0; channel < nchannels; channel++) {
			for (sample = 0; sample < nsamples; sample++) {
				MAT3D(c_data, sample, channel, trial, nsamples, nchannels) =
					*(double *)PyArray_GETPTR3(data, trial, channel, sample);
			}
		}
	}

	double * c_results = malloc(nchannels * ntrials * sizeof(double));

	int all_trial_result = do_process_all_trials(c_data, nsamples, nchannels, ntrials, c_results, nbins, nthreads);

	if (all_trial_result != 0) {
		PyErr_SetString(ompkError, "Unable to compute Komplexity ");
	}





	int result_ndims = 2;
	npy_intp * result_dims = malloc(result_ndims * sizeof(npy_intp));

	result_dims[0] = nchannels;
	result_dims[1] = ntrials;
	PyArrayObject *py_result = (PyArrayObject *) PyArray_ZEROS(result_ndims, result_dims, NPY_DOUBLE, 0); //CTYPE Zeros array

	for (trial = 0; trial < ntrials; trial++) {
		for (channel = 0; channel < nchannels; channel++) {
			*(double *)PyArray_GETPTR2(py_result, channel, trial) =
				MAT2D(c_results, channel, trial, nchannels);
		}
	}
	free(c_data);
	free(c_results);
	free(result_dims);

	PyObject * retorno = (PyObject *) py_result;
	return retorno;
}

static PyMethodDef OmpkMethods[] = {
	{"komplexity",  ompk_komplexity, METH_VARARGS, "The Komplexity of the given matrix across second dimension"},
	{NULL, NULL, 0, NULL}
};


#if PY_MAJOR_VERSION == 3
static struct PyModuleDef ompkmodule = {
   PyModuleDef_HEAD_INIT,
   "ompk",   /* name of module */
   NULL, /* module documentation, may be NULL */
   -1,       /* size of per-interpreter state of the module,
                or -1 if the module keeps state in global variables. */
   OmpkMethods
};
#endif

#if PY_MAJOR_VERSION == 3
PyMODINIT_FUNC PyInit_ompk(void) {
	PyObject *m;
	m = PyModule_Create(&ompkmodule);
	if (m == NULL)
		return NULL;
#else
PyMODINIT_FUNC initompk(void) {
	PyObject *m;
	m = Py_InitModule("ompk", OmpkMethods);
	if (m == NULL)
		return;
#endif

	ompkError = PyErr_NewException("ompk.error", NULL, NULL);
	Py_INCREF(ompkError);
	PyModule_AddObject(m, "error", ompkError);
	import_array();
#if PY_MAJOR_VERSION == 3
	return m;
#endif

}

#endif