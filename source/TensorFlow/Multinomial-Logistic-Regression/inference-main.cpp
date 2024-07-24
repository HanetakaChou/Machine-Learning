#include <tensorflow/lite/c/c_api.h>
#include <tensorflow/lite/delegates/gpu/delegate.h>
#include <stdint.h>
#include <assert.h>
#include <stdio.h>
#include <cstring>
#include <openblas/cblas.h>
#include <iostream>

static void default_error_reporter(void *, const char *format, va_list args);

int main()
{
        // Data
        constexpr int const m = 7;
        constexpr int const n = 28 * 28;

        constexpr float const X_test[m][n] = {
            {
#include "MNIST_X_test_0.inl"
            },
            {
#include "MNIST_X_test_1.inl"
            },
            {
#include "MNIST_X_test_2.inl"
            },
            {
#include "MNIST_X_test_3.inl"
            },
            {
#include "MNIST_X_test_4.inl"
            },
            {
#include "MNIST_X_test_5.inl"
            },
            {
#include "MNIST_X_test_6.inl"
            }};

        constexpr int const y_test[m] = {
#include "MNIST_y_test_0.inl"
            ,
#include "MNIST_y_test_1.inl"
            ,
#include "MNIST_y_test_2.inl"
            ,
#include "MNIST_y_test_3.inl"
            ,
#include "MNIST_y_test_4.inl"
            ,
#include "MNIST_y_test_5.inl"
            ,
#include "MNIST_y_test_6.inl"
        };

        // Model
        TfLiteModel *tflite_model = NULL;
        {
                // NOTE: the memory of the "model_data" must remain valid as long as the "TfLiteModel" is still in use.
                // We use "static" keyword for convenience
                static uint8_t tflite_model_data[] = {
#include "Multinomial-Logistic-Regression.inl"
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
        for (int i = 0; i < m; ++i)
        {
                std::memcpy(tflite_input, X_test[i], sizeof(float) * n);

                TfLiteStatus tflite_status_invoke = TfLiteInterpreterInvoke(tflite_interpreter);
                assert(kTfLiteOk == tflite_status_invoke);

                float *logit_prediction = tflite_output;
                int prediction = cblas_ismax(10, logit_prediction, 1);
                std::cout << "Prediction: " << prediction << " Target: " << y_test[i] << std::endl;
        }

        TfLiteInterpreterDelete(tflite_interpreter);
        TfLiteGpuDelegateV2Delete(tflite_delegate);
        TfLiteModelDelete(tflite_model);

        return 0;
}

static void default_error_reporter(void *, const char *format, va_list args)
{
        vprintf(format, args);
        return;
}