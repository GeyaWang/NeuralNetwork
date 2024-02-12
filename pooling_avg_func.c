#include <Python.h>
#include <numpy/arrayobject.h>
#include <string.h>
#include <stdlib.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))


static PyObject *_forward(PyObject* self, PyObject *args) {
    PyObject *X_obj;
    PyObject *output_shape_tuple;
    PyObject *pool_size_tuple;
    PyObject *strides_tuple;
    PyObject *pad_tuple;

    int H2; int W2; int C;
    int pool_x; int pool_y;
    int stride_x; int stride_y;
    int pad_x; int pad_y;

    PyArg_ParseTuple(args, "OOOOO", &X_obj, &output_shape_tuple, &pool_size_tuple, &strides_tuple, &pad_tuple);
    PyArg_ParseTuple(output_shape_tuple, "iii", &H2, &W2, &C);
    PyArg_ParseTuple(pool_size_tuple, "ii", &pool_x, &pool_y);
    PyArg_ParseTuple(strides_tuple, "ii", &stride_x, &stride_y);
    PyArg_ParseTuple(pad_tuple, "ii", &pad_x, &pad_y);

    PyObject *X = PyArray_FROM_OTF(X_obj, NPY_FLOAT64, NPY_ARRAY_IN_ARRAY);

    int N = PyArray_DIM(X, 0);
    int H1 = PyArray_DIM(X, 1);
    int W1 = PyArray_DIM(X, 2);

    npy_intp dims[] = {N, H2, W2, C};
    PyObject *Y = PyArray_SimpleNew(4, dims, NPY_DOUBLE);

    double *X_data = (double *)PyArray_DATA(X);
    double *Y_data = (double *)PyArray_DATA(Y);

    int pool_size = pool_x * pool_y;
    double sum;

    for (int n = 0; n < N; ++n) {
        for (int h = 0; h < H2; ++h) {
            for (int w = 0; w < W2; ++w) {
                for (int c = 0; c < C; ++c) {

                    for (int i = MAX(0, pad_x - h * stride_x); i < pool_x; ++i) {
                        for (int j = MAX(0, pad_y - w * stride_y); j < pool_y; ++j) {
                            sum += X_data[(((n * H1) + (h * stride_x + i - pad_x)) * W1 + (w * stride_y + j - pad_y)) * C + c];
                        }
                    }
                    Y_data[(((n * H2) + h) * W2 + w) * C + c] = sum / pool_size;
                }
            }
        }
    }

    Py_DECREF(X);

    return Y;
}

static PyObject *_backward(PyObject* self, PyObject *args) {
    PyObject *X_obj;
    PyObject *dY_obj;
    PyObject *pool_size_tuple;
    PyObject *strides_tuple;
    PyObject *pad_tuple;

    int pool_x; int pool_y;
    int stride_x; int stride_y;
    int pad_x; int pad_y;

    PyArg_ParseTuple(args, "OOOOO", &X_obj, &dY_obj, &pool_size_tuple, &strides_tuple, &pad_tuple);
    PyArg_ParseTuple(pool_size_tuple, "ii", &pool_x, &pool_y);
    PyArg_ParseTuple(strides_tuple, "ii", &stride_x, &stride_y);
    PyArg_ParseTuple(pad_tuple, "ii", &pad_x, &pad_y);

    PyObject *X = PyArray_FROM_OTF(X_obj, NPY_FLOAT64, NPY_ARRAY_IN_ARRAY);
    PyObject *dY = PyArray_FROM_OTF(dY_obj, NPY_FLOAT64, NPY_ARRAY_IN_ARRAY);

    int N = PyArray_DIM(X, 0);
    int H1 = PyArray_DIM(X, 1);
    int W1 = PyArray_DIM(X, 2);
    int C = PyArray_DIM(X, 3);
    int H2 = PyArray_DIM(dY, 1);
    int W2 = PyArray_DIM(dY, 2);

    npy_intp dims[] = {N, H1, W1, C};
    PyObject *dX = PyArray_SimpleNew(4, dims, NPY_DOUBLE);
    PyArray_FILLWBYTE(dX, 0);

    double *X_data = (double *)PyArray_DATA(X);
    double *dY_data = (double *)PyArray_DATA(dY);
    double *dX_data = (double *)PyArray_DATA(dX);

    int pool_size = pool_x * pool_y;
    double val;

    for (int n = 0; n < N; ++n) {
        for (int h = 0; h < H2; ++h) {
            for (int w = 0; w < W2; ++w) {
                for (int c = 0; c < C; ++c) {

                    val = dY_data[(((n * H2) + h) * W2 + w) * C + c] / pool_size;

                    for (int i = MAX(0, pad_x - h * stride_x); i < pool_x; ++i) {
                        for (int j = MAX(0, pad_y - w * stride_y); j < pool_y; ++j) {
                            dX_data[(((n * H1) + (h * stride_x + i - pad_x)) * W1 + (w * stride_y + j - pad_y)) * C + c] = val;
                        }
                    }
                }
            }
        }
    }

    Py_DECREF(X);
    Py_DECREF(dY);

    return dX;
}

static struct PyMethodDef methods[] = {
    {"forward", _forward, METH_VARARGS, "Forward average pooling step"},
    {"backward", _backward, METH_VARARGS, "Backward average pooling step"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef pooling_avg_func = {
    PyModuleDef_HEAD_INIT,
    "pooling_avg_func",
    NULL,
    -1,
    methods
};

PyMODINIT_FUNC PyInit_pooling_avg_func(void) {
    PyObject *module = PyModule_Create(&pooling_avg_func);
    import_array();
    return module;
}

