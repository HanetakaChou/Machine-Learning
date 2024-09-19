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
    constexpr int const kappa = 15;

    constexpr int32_t const data_user_ids[kappa] = {0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3};

    constexpr int32_t const data_item_ids[kappa] = {0, 1, 3, 4, 0, 2, 3, 4, 0, 2, 3, 4, 0, 1, 3};

    constexpr float const data_interactions[kappa] = {5, 5, 0, 0, 5, 4, 0, 0, 0, 0, 5, 5, 0, 0, 4};

    // Model
    TfLiteModel *tflite_model = NULL;
    {
        // NOTE: the memory of the "model_data" must remain valid as long as the "TfLiteModel" is still in use.
        // We use "static" keyword for convenience
        static uint8_t tflite_model_data[] = {
#include "regression-based-collaborative-filtering.inl"
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

    int32_t *tflite_input_0 = TfLiteInterpreterGetInputTensor(tflite_interpreter, 0)->data.i32;
    int32_t *tflite_input_1 = TfLiteInterpreterGetInputTensor(tflite_interpreter, 1)->data.i32;
    float *tflite_output = TfLiteInterpreterGetOutputTensor(tflite_interpreter, 0)->data.f;

    // Inference
    for (int i = 0; i < kappa; ++i)
    {
        tflite_input_0[0] = data_user_ids[i];
        tflite_input_1[0] = data_item_ids[i];

        TfLiteStatus tflite_status_invoke = TfLiteInterpreterInvoke(tflite_interpreter);
        assert(kTfLiteOk == tflite_status_invoke);

        float prediction = tflite_output[0];
        std::cout << "Prediction: " << prediction << " Target: " << data_interactions[i] << std::endl;
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