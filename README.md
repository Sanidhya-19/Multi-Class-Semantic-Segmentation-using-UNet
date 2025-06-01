This project implements a multi-class image segmentation model using the UNet architecture. The model segments eye images into iris, pupil, and sclera regions. The same model is exported and benchmarked using Keras (.h5), ONNX (.onnx), and TFLite (.tflite) formats to compare performance and deployment readiness.

📁 Dataset
The dataset used is a custom eye segmentation dataset structured in PASCAL VOC style.

arduino
Copy
Edit
/dataset/
    ├── train/
    │   ├── image/
    │   └── mask/
    ├── val/
    │   ├── image/
    │   └── mask/
Each mask is a 2D image where:

0 = background

1 = iris

2 = pupil

3 = sclera

🏗 Model Architecture
The model follows the UNet architecture:

Encoder with downsampling convolutions

Bottleneck

Decoder with upsampling and skip connections

Final 1×1 convolution with softmax for multi-class output

Supported input shape: (256, 256, 3)
Output shape: (256, 256, 4) (4 classes)

🧪 Inference Pipelines
1. Keras (.h5)
python
Copy
Edit
model = tf.keras.models.load_model("unet_multiclass_final_model.h5")
pred = model.predict(input_tensor)
mask = np.argmax(pred[0], axis=-1)
2. ONNX (.onnx)
python
Copy
Edit
import onnxruntime as ort
session = ort.InferenceSession("unet.onnx")
output = session.run(None, {input_name: input_tensor})[0]
mask = np.argmax(output[0], axis=-1)
3. TFLite (.tflite)
python
Copy
Edit
interpreter = tf.lite.Interpreter(model_path="unet.tflite")
interpreter.allocate_tensors()
interpreter.set_tensor(input_index, input_tensor)
interpreter.invoke()
output = interpreter.get_tensor(output_index)
mask = np.argmax(output[0], axis=-1)
🚀 Exported Models
Format	File	Use Case
Keras	unet_multiclass_final_model.h5	Training & validation (TensorFlow)
ONNX	unet.onnx	Cross-platform inference (C++, C#, etc.)
TFLite	unet.tflite	Mobile & edge deployment
PyTorch	best_unet_model.pth	Research, fine-tuning, and export

⚡ Performance Benchmark
Measured on a sample image of size (256, 256):

Format	Inference Time	Notes
Keras	~20–30 ms	Baseline
ONNX	~10–20 ms	Faster on CPU
TFLite	~5–10 ms	Optimized for edge/mobile

Pixel-wise mask agreement between Keras and ONNX: ~99.8%

📊 Visualization
Input Image	Iris Mask	Pupil Mask	Sclera Mask
			

Masks are visualized as grayscale outputs (white = segmented region).

🛠 Setup Instructions
bash
Copy
Edit
# Install dependencies
pip install tensorflow onnxruntime opencv-python matplotlib

# For TFLite
# Option 1: Use tf.lite.Interpreter from TensorFlow
# Option 2: Install separately (for edge devices)
# pip install tflite-runtime
📝 Project Structure
graphql
Copy
Edit
.
├── model.py                      # UNet architecture (PyTorch)
├── keras_inference.py           # Inference with .h5 model
├── onnx_inference.py            # Inference with ONNX
├── tflite_inference.py          # Inference with TFLite
├── convert_to_onnx.py           # Keras to ONNX export
├── convert_to_tflite.py         # Keras to TFLite export
├── compare_models.py            # Compare inference times
├── best_unet_model.pth          # PyTorch model
├── unet_multiclass_final_model.h5
├── unet.onnx
├── unet.tflite
└── README.md
📌 Notes
Ensure image inputs are normalized to [0, 1].

Inference outputs are post-processed with argmax to convert softmax outputs into class labels.

Use quantization (optional) for optimizing TFLite model size and performance.
