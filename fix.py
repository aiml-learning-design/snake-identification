# metal_test.py
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress logs
os.environ['TF_METAL_ENABLED'] = '1'

# Critical isolation technique
import ctypes
try:
    metal = ctypes.CDLL('/System/Library/Frameworks/Metal.framework/Metal')
    metal.MTLCreateSystemDefaultDevice()
except:
    pass  # Continue even if Metal fails

import tensorflow as tf
print(f"TensorFlow {tf.__version__} loaded")
print("Devices:", [d.device_type for d in tf.config.list_physical_devices()])