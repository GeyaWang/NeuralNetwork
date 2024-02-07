#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
#include <string.h>
#include <stdlib.h>

static PyObject *_forward(PyObject* self, PyObject *args) {
    PyArrayObject *X;
    PyArrayObject *K;
    PyArrayObject *B;
    char *padding;
    PyArg_ParseTuple(args, "OOOs", &X, &K, &B, &padding);

    // Get input dimensions
    int N = PyArray_DIM(X, 0);
    int H1 = PyArray_DIM(X, 1);
    int W1 = PyArray_DIM(X, 2);
    int k1 = PyArray_DIM(K, 0);
    int k2 = PyArray_DIM(K, 1);
    int C1 = PyArray_DIM(K, 2);
    int C2 = PyArray_DIM(K, 3);

    int pad_x; int pad_y;
    int H2; int W2;

    if (strcmp(padding, "same") == 0) {
        pad_x = k1 / 2; pad_y = k2 / 2;
        H2 = H1; W2 = W1;
    } else {  // "valid"
        pad_x = 0; pad_y = 0;
        H2 = H1 - k1 + 1; W2 = W1 - k2 + 1;
    }

    // Initialise output
    npy_intp dims[] = {N, H2, W2, C2};
    PyArrayObject *Y = PyArray_SimpleNew(4, dims, NPY_DOUBLE);

    double *X_data = (double*)PyArray_DATA(X);
    double *K_data = (double*)PyArray_DATA(K);
    double *B_data = (double*)PyArray_DATA(B);
    double *Y_data = (double*)PyArray_DATA(Y);

    double sum;
    double bias;

    // Perform convolution
    for (int c2 = 0; c2 < C2; ++c2) {
        bias = B_data[c2];

        for (int n = 0; n < N; ++n) {
            for (int h = 0; h < H2; ++h) {
                for (int w = 0; w < W2; ++w) {

                    sum = 0;
                    for (int c1 = 0; c1 < C1; ++c1) {
                        for (int i = max(0, pad_x - h); i < min(k1, H1 + pad_x - h); ++i) {
                            for (int j = max(0, pad_y - w); j < min(k2, W1 + pad_y - w); ++j) {
                                sum += X_data[(((n * H1) + (h + i - pad_x)) * W1 + (w + j - pad_y)) * C1 + c1] * K_data[(((i * k2) + j) * C1 + c1) * C2 + c2];
                            }
                        }
                    }
                    Y_data[(((n * H2) + h) * W2 + w) * C2 + c2] = bias + sum;
                }
            }
        }
    }

    return PyArray_Return(Y);
}

static PyObject *_backward(PyObject* self, PyObject *args) {
    PyArrayObject *X;
    PyArrayObject *K;
    PyArrayObject *dY;
    char *padding;
    PyArg_ParseTuple(args, "OOOs", &X, &K, &dY, &padding);

    // Get input dimensions
    int N = PyArray_DIM(X, 0);
    int H1 = PyArray_DIM(X, 1);
    int W1 = PyArray_DIM(X, 2);
    int k1 = PyArray_DIM(K, 0);
    int k2 = PyArray_DIM(K, 1);
    int C1 = PyArray_DIM(K, 2);
    int C2 = PyArray_DIM(K, 3);
    int H2 = PyArray_DIM(dY, 1);
    int W2 = PyArray_DIM(dY, 2);

    // Initialise error gradients
    npy_intp dims_dX[] = {N, H1, W1, C1};
    npy_intp dims_dW[] = {k1, k2, C1, C2};
    npy_intp dims_dB[] = {C2};
    PyArrayObject *dX = PyArray_SimpleNew(4, dims_dX, NPY_DOUBLE);
    PyArrayObject *dW = PyArray_SimpleNew(4, dims_dW, NPY_DOUBLE);
    PyArrayObject *dB = PyArray_SimpleNew(1, dims_dB, NPY_DOUBLE);

    // fill with zeros
    PyArray_FILLWBYTE(dX, 0);
    PyArray_FILLWBYTE(dW, 0);

    double *X_data = (double*)PyArray_DATA(X);
    double *K_data = (double*)PyArray_DATA(K);
    double *dX_data = (double*)PyArray_DATA(dX);
    double *dW_data = (double*)PyArray_DATA(dW);
    double *dB_data = (double*)PyArray_DATA(dB);
    double *dY_data = (double*)PyArray_DATA(dY);

    int pad_dX_x; int pad_dX_y;
    int pad_dW_x; int pad_dW_y;

    if (strcmp(padding, "same") == 0) {
        pad_dX_x = k1 / 2; pad_dX_y = k2 / 2;
        pad_dW_x = k1 / 2; pad_dW_y = k2 / 2;
    } else {  // "valid"
        pad_dX_x = k1 - 1; pad_dX_y = k2 - 1;
        pad_dW_x = 0; pad_dW_y = 0;
    }

    int ki = k1 - 1; int kj = k2 - 1;
    int n_H1; int n_H2;

    int max_x; int max_y;
    double dB_sum; double dW_sum; double dX_sum;
    double K_val;

    // dB
    for (int c2 = 0; c2 < C2; ++c2) {

        dB_sum = 0;
        for (int n = 0; n < N; ++n) {

            n_H1 = n * H1;
            n_H2 = n * H2;

            for (int h2 = 0; h2 < H2; ++h2) {
                for (int w2 = 0; w2 < W2; ++w2) {
                    dB_sum += dY_data[((n_H2 + h2) * W2 + w2) * C2 + c2];
                }
            }

            // dX
            for (int c1 = 0; c1 < C1; ++c1) {
                for (int i = 0; i < k1; ++i) {
                    for (int j = 0; j < k2; ++j) {

                        max_x = min(H1, H2 + pad_dX_x - i);
                        max_y = min(W1, W2 + pad_dX_y - j);
                        K_val = K_data[((((ki - i) * k2) + (kj - j)) * C1 + c1) * C2 + c2];
                        dX_sum = 0;
                        for (int h1 = max(0, pad_dX_x - i); h1 < max_x; ++h1) {
                            for (int w1 = max(0, pad_dX_y - j); w1 < max_y; ++w1) {
                                dX_sum += dY_data[((n_H2 + (h1 + i - pad_dX_x)) * W2 + (w1 + j - pad_dX_y)) * C2 + c2] * K_val;
                            }
                        }
                        dX_data[((n_H1 + h1) * W1 + w1) * C1 + c1] += dX_sum;

                        // dW
                        max_x = min(H2, H1 + pad_dW_x - i);
                        max_y = min(W2, W1 + pad_dW_y - j);
                        dW_sum = 0;
                        for (int h2 = max(0, pad_dW_x - i); h2 < max_x; ++h2) {
                            for (int w2 = max(0, pad_dW_y - j); w2 < max_y; ++w2) {
                                dW_sum += X_data[((n_H1 + (h2 + i - pad_dW_x)) * W1 + (w2 + j - pad_dW_y)) * C1 + c1] * dY_data[((n_H2 + h2) * W2 + w2) * C2 + c2];
                            }
                        }
                        dW_data[(((i * k2) + j) * C1 + c1) * C2 + c2] += dW_sum;
                    }
                }
            }
        }
        dB_data[c2] = dB_sum;
    }

    return Py_BuildValue("OOO", dX, dW, dB);
}


static struct PyMethodDef methods[] = {
    {"forward", _forward, METH_VARARGS, "Forward convolution step"},
    {"backward", _backward, METH_VARARGS, "Backward convolution step"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef conv_func = {
    PyModuleDef_HEAD_INIT,
    "conv_func",
    NULL,
    -1,
    methods
};

PyMODINIT_FUNC PyInit_conv_func(void) {
    PyObject *module = PyModule_Create(&conv_func);
    import_array();
    return module;
}
