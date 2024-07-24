#include <tensorflow/lite/c/c_api.h>
#include <tensorflow/lite/delegates/gpu/delegate.h>
#include <stdint.h>
#include <assert.h>
#include <stdio.h>
#include <iostream>

static void default_error_reporter(void *, const char *format, va_list args);

int main()
{
    // Data
    constexpr int const m = 4;
    constexpr int const n = 2;

    constexpr float const X[m * n] = {
        10.0, 52.0,
        2.0, 73.0,
        5.0, 55.0,
        12.0, 49.0};

    constexpr float const y[m] = {
        1.0,
        0.0,
        0.0,
        1.0};

    // Model
    TfLiteModel *tflite_model = NULL;
    {
        // NOTE: the memory of the "model_data" must remain valid as long as the "TfLiteModel" is still in use.
        // We use "static" keyword for convenience
        static uint8_t tflite_model_data[] = {
#include "logistic-regression.inl"
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
        tflite_input[0] = X[n * i];
        tflite_input[1] = X[n * i + 1];

        TfLiteStatus tflite_status_invoke = TfLiteInterpreterInvoke(tflite_interpreter);
        assert(kTfLiteOk == tflite_status_invoke);

        float logit_prediction = tflite_output[0];
        int prediction = (logit_prediction > 0.0F);
        std::cout << "Prediction: " << prediction << " Target: " << y[i] << std::endl;
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