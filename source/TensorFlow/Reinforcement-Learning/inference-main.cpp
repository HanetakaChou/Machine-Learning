#include <tensorflow/lite/c/c_api.h>
#include <tensorflow/lite/delegates/gpu/delegate.h>
#include <Python.h>
#include <stdint.h>
#include <assert.h>
#include <stdio.h>
#include <openblas/cblas.h>
#include <vector>
#include <iostream>

static void default_error_reporter(void *, const char *format, va_list args);

int main()
{
    // Data

    Py_Initialize();

    PyObject *py_env = NULL;
    {
        PyObject *py_gym = PyImport_ImportModule("gymnasium");
        assert(py_gym);

        PyObject *py_make = PyObject_GetAttrString(py_gym, "make");
        assert(py_make);

        PyObject *py_id = PyUnicode_FromString("LunarLander-v2");
        assert(py_id);

        PyObject *py_args = PyTuple_Pack(1, py_id);
        assert(py_args);

        PyObject *py_render_mode = PyUnicode_FromString("human");
        assert(py_render_mode);

        PyObject *py_kwargs = PyDict_New();
        assert(py_kwargs);

        PyDict_SetItemString(py_kwargs, "render_mode", py_render_mode);

        py_env = PyObject_Call(py_make, py_args, py_kwargs);
        PyErr_Print();

        Py_DECREF(py_kwargs);

        Py_DECREF(py_render_mode);

        Py_DECREF(py_args);

        Py_DECREF(py_id);

        Py_DECREF(py_make);

        Py_DECREF(py_gym);
    }
    assert(py_env);

    int state_count = -1;
    {
        PyObject *py_observation_space = PyObject_GetAttrString(py_env, "observation_space");
        assert(py_observation_space);

        PyObject *py_shape = PyObject_GetAttrString(py_observation_space, "shape");
        assert(py_shape);

        PyObject *py_state_size = PyTuple_GetItem(py_shape, 0);

        state_count = PyLong_AsLong(py_state_size);

        Py_DECREF(py_shape);

        Py_DECREF(py_observation_space);
    }
    assert(state_count > 0);

    int action_count = -1;
    {
        PyObject *py_action_space = PyObject_GetAttrString(py_env, "action_space");
        assert(py_action_space);

        PyObject *py_action_count = PyObject_GetAttrString(py_action_space, "n");
        assert(py_action_count);

        action_count = PyLong_AsLong(py_action_count);

        Py_DECREF(py_action_count);

        Py_DECREF(py_action_space);
    }
    assert(action_count > 0);

    constexpr int const maximum_step_count = 1000;

    // Model
    TfLiteModel *tflite_model = NULL;
    {
        // NOTE: the memory of the "model_data" must remain valid as long as the "TfLiteModel" is still in use.
        // We use "static" keyword for convenience
        static uint8_t tflite_model_data[] = {
#include "q-network.inl"
        };

        tflite_model = TfLiteModelCreateWithErrorReporter(tflite_model_data, sizeof(tflite_model_data), default_error_reporter, NULL);
    }
    assert(tflite_model);

    TfLiteDelegate *tflite_delegate = NULL;
    {
        TfLiteGpuDelegateOptionsV2 tflite_delegate_options = TfLiteGpuDelegateOptionsV2Default();
        tflite_delegate_options.experimental_flags = TFLITE_GPU_EXPERIMENTAL_FLAGS_NONE;

        tflite_delegate = TfLiteGpuDelegateV2Create(&tflite_delegate_options);
    }
    assert(tflite_delegate);

    TfLiteInterpreter *tflite_interpreter = NULL;
    {
        TfLiteInterpreterOptions *tflite_interpreter_options = TfLiteInterpreterOptionsCreate();
        assert(tflite_interpreter_options);

        TfLiteInterpreterOptionsAddDelegate(tflite_interpreter_options, tflite_delegate);

        tflite_interpreter = TfLiteInterpreterCreate(tflite_model, tflite_interpreter_options);
        assert(tflite_interpreter);

        TfLiteInterpreterOptionsDelete(tflite_interpreter_options);
    }

    TfLiteStatus tflite_status_allocate_tensors = TfLiteInterpreterAllocateTensors(tflite_interpreter);
    assert(kTfLiteOk == tflite_status_allocate_tensors);

    float *tflite_input = TfLiteInterpreterGetInputTensor(tflite_interpreter, 0)->data.f;
    float *tflite_output = TfLiteInterpreterGetOutputTensor(tflite_interpreter, 0)->data.f;

    // Inference
    std::vector<float> cxx_state(static_cast<size_t>(state_count));
    {
        PyObject *py_reset_return = PyObject_CallMethod(py_env, "reset", NULL);
        assert(py_reset_return);

        PyObject *py_state = PyTuple_GetItem(py_reset_return, 0);

        PyObject *py_state_list = PySequence_List(py_state);
        assert(py_state_list);

        assert(PyList_Size(py_state_list) == state_count);

        for (int state_index = 0; state_index < state_count; ++state_index)
        {
            PyObject *py_item = PyList_GetItem(py_state_list, state_index);
            cxx_state[state_index] = PyFloat_AsDouble(py_item);
        }

        Py_DECREF(py_state_list);

        Py_DECREF(py_reset_return);
    }

    for (int step_index = 0; step_index < maximum_step_count; ++step_index)
    {
        for (int state_index = 0; state_index < state_count; ++state_index)
        {
            tflite_input[state_index] = cxx_state[state_index];
        }

        TfLiteStatus tflite_status_invoke = TfLiteInterpreterInvoke(tflite_interpreter);
        assert(kTfLiteOk == tflite_status_invoke);

        int action = cblas_ismax(action_count, tflite_output, 1);

        bool done = false;
        {
            PyObject *py_action = PyLong_FromLong(action);
            assert(py_action);
            PyErr_Print();

            PyObject *py_step_return = PyObject_CallMethod(py_env, "step", "O", py_action);
            PyErr_Print();

            assert(py_step_return);

            Py_DECREF(py_action);

            PyObject *py_next_state = PyTuple_GetItem(py_step_return, 0);

            PyObject *py_next_state_list = PySequence_List(py_next_state);
            assert(py_next_state_list);

            assert(PyList_Size(py_next_state_list) == state_count);

            for (int state_index = 0; state_index < state_count; ++state_index)
            {
                PyObject *py_item = PyList_GetItem(py_next_state_list, state_index);
                cxx_state[state_index] = PyFloat_AsDouble(py_item);
            }

            Py_DECREF(py_next_state_list);

            PyObject *py_done = PyTuple_GetItem(py_step_return, 2);
            done = PyObject_IsTrue(py_done);

            Py_DECREF(py_step_return);
        }

        {
            PyObject *py_render_return = PyObject_CallMethod(py_env, "render", NULL);
            assert(py_render_return);

            Py_DECREF(py_render_return);
        }

        if (done)
        {
            break;
        }
    }

    {
        PyObject *py_close_return = PyObject_CallMethod(py_env, "close", NULL);
        assert(py_close_return);

        Py_DECREF(py_close_return);
    }

    TfLiteInterpreterDelete(tflite_interpreter);
    TfLiteGpuDelegateV2Delete(tflite_delegate);
    TfLiteModelDelete(tflite_model);

    Py_DECREF(py_env);
    Py_Finalize();

    return 0;
}

static void default_error_reporter(void *, const char *format, va_list args)
{
    vprintf(format, args);
    return;
}