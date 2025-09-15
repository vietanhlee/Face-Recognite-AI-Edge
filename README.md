# Face Recognition System

## Introduction

This project provides a robust and efficient face recognition system designed for real-time applications.  
It leverages a **quantized YOLO model (OpenVINO INT8)** for fast and accurate face detection, and a **quantized Facenet model (TFLite INT8)** to extract high-quality face embeddings for recognition.  
The system is optimized for both speed and accuracy, making it suitable for deployment on edge devices and in resource-constrained environments.

### Main Features
- **Real-time Face Detection and Recognition:** Instantly detect and identify faces from webcam or video streams using YOLO and Facenet.  
- **Face Registration:** Add new faces to the recognition database with a unique name or label.  
- **Update and Edit:** Modify or update the information and embeddings of registered faces.  
- **Delete Faces:** Remove faces from the recognition database as needed.  
- **Database Management:** Efficiently manage, search, and update the face embeddings for scalable recognition.  

The solution is modular, easy to extend, and can be integrated into larger security, attendance, or access control systems.

---

## Installation

### Requirements
- Python >= 3.10

Install all required dependencies:
```bash
pip install -r requirements.txt
````

---

## Usage

All functionalities are available via **command line (CMD)**:

* **Run the main face recognition system**

  ```bash
  python main.py
  ```

* **Register a new face**

  ```bash
  python register_face.py --name "Person_Name" --id "ID"
  ```

* **Delete a registered face**

  ```bash
  python delete_face.py --id "ID"
  ```

* **Create or update the vector database**

  ```bash
  python create_vt_db.py --reinit "0 or 1"
  ```

---

## Example 

1. Register a new face:

   ```bash
   python register_face.py --name "Alice" --id 11
   ```

2. Update the vector database:

   ```bash
   python create_vt_db.py --reinit 1
   ```

3. Run the recognition system:

   ```bash
   python main.py
   ```

4. Delete a face if needed:

   ```bash
   python delete_face.py --id 1
   ```

---

## References

* [YOLO](https://github.com/ultralytics/ultralytics)
* [Facenet](https://github.com/davidsandberg/facenet)
* [OpenVINO](https://docs.openvino.ai/)
* [Tensorflow Lite (TFLite)](https://www.tensorflow.org/lite)
