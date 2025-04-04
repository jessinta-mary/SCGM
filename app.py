import streamlit as st
import asyncio
from bleak import BleakClient, BleakScanner
import joblib
import numpy as np
import onnxruntime as ort  # ONNX runtime for inference

# Load the trained scaler (needed to normalize real sensor data)
scaler = joblib.load("scaler.pkl")  # ✅ Load the same scaler used during training

# Load the trained ONNX model path from the pickle file
model_path = joblib.load("glucose_prediction_model.pkl")

# Load ONNX model using ONNX Runtime
session = ort.InferenceSession(model_path)

# Get correct input name from the ONNX model
input_name = session.get_inputs()[0].name  # ✅ Fetch the actual input name

st.title("Continuous Glucose Monitoring")
st.subheader("Live data from ESP32")

# ESP32 BLE UUIDs (Must match ESP32 code)
SERVICE_UUID = "12345678-1234-5678-1234-56789abcdef0"
CHARACTERISTIC_UUID = "abcd1234-5678-1234-5678-abcdef123456"

# Function to connect and communicate with ESP32
async def connect_ble():
    st.write("Scanning for ESP32...")

    # Automatically find ESP32 device
    devices = await BleakScanner.discover()
    esp_address = None

    for device in devices:
        if device.name and "ESP32_CGM" in device.name:  # ✅ Ensure device.name is not None
            esp_address = device.address
            break

    if not esp_address:
        st.error("ESP32 device not found!")
        return None

    st.write(f"Connecting to ESP32 at {esp_address}...")

    async with BleakClient(esp_address) as client:
        if await client.is_connected():
            st.success("Connected to ESP32!")

            # Read sensor data from ESP32
            raw_data = await client.read_gatt_char(CHARACTERISTIC_UUID)
            sensor_data = raw_data.decode("utf-8").strip()
            st.write("Received Sensor Data:", sensor_data)

            # Process data with ONNX model
            if sensor_data:
                try:
                    # ✅ Convert sensor data to float array
                    params = np.array([float(x) for x in sensor_data.split(",")], dtype=np.float32).reshape(1, -1)
                    
                    # ✅ Normalize the data using the saved scaler
                    params_scaled = scaler.transform(params)
                    
                    # ✅ Reshape for CNN-LSTM (batch_size=1, timesteps=features, channels=1)
                    params_scaled = params_scaled.reshape(1, params_scaled.shape[1], 1)

                    # ✅ Use correct input name for ONNX inference
                    glucose_level = session.run(None, {input_name: params_scaled})[0][0][0]
                    st.write(f"Predicted Glucose Level: {glucose_level:.2f} mg/dL")

                    # Send back glucose level to ESP32
                    await client.write_gatt_char(CHARACTERISTIC_UUID, str(glucose_level).encode("utf-8"))

                    # Store past data
                    with open("glucose_data.txt", "a") as f:
                        f.write(f"{sensor_data},{glucose_level}\n")
                
                except Exception as e:
                    st.error(f"Prediction Error: {e}")

# Button to start monitoring
if st.button("Start Monitoring"):
    asyncio.run(connect_ble())