
# CerebroClassify 🧠

An advanced web application that utilizes a deep learning model to classify brain MRI scans into four categories: **Glioma**, **Meningioma**, **Pituitary Tumor**, and **Healthy**. The app provides an intuitive interface for users to upload MRI images and receive predictions with confidence scores.

![Gemini_Generated_Image](https://github.com/user-attachments/assets/dafab97b-2be0-4f17-be24-1aa2144ec561)

## **Table of Contents**

- [Features](#features)
- [Demo](#demo)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Model Details](#model-details)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [License](#license)
- [Disclaimer](#disclaimer)
- [Acknowledgements](#acknowledgements)

## **Features**

- **User-Friendly Interface**: Simple and elegant UI built with Streamlit.
- **Real-Time Prediction**: Upload an MRI image and get instant predictions.
- **Detailed Results**:
  - Displays the predicted class with confidence percentage.
  - Provides a bar chart of confidence scores for all classes.
  - Offers explanations and links to learn more about each condition.
- **Downloadable Report**: Option to download a text report of the prediction and confidence scores.
- **Mobile Responsive**: The app is optimized for use on various devices.

## **Demo**

Try the live demo: **[CerebroClassify Live Demo](https://your-demo-link.com)**

*(Replace the above link with the actual URL if you deploy the app)*

## **Installation**

### **Prerequisites**

- Python 3.7 - 3.10
- Git installed on your local machine

### **Create a Virtual Environment**

It's recommended to use a virtual environment to manage dependencies.

```bash
python -m venv venv
```

Activate the virtual environment:

On Windows:

```bash
venv\Scriptsctivate
```

On macOS/Linux:

```bash
source venv/bin/activate
```

### **Install Dependencies**

```bash
pip install -r requirements.txt
```

### **Download the Model File**

Place your trained model file `brain_tumor_classifier.h5` in the project directory. Ensure that the model file is compatible with the code.

## **Usage**

### **Run the Application**

```bash
streamlit run app.py
```

### **Using the App**

- Open your web browser and navigate to the URL provided in the terminal (usually [http://localhost:8501](http://localhost:8501)).
- Upload an MRI image in JPG or PNG format.
- Click on **Predict** to get the classification results.
- View detailed explanations and confidence scores.
- Download the report if you wish to save the results.

## **Project Structure**

```markdown
BrainTumorAI/
├── app.py
├── style.css
├── brain_tumor_classifier.h5
├── requirements.txt
├── README.md
└── images/
    └── screenshot.png
```

- `app.py`: Main application script.
- `style.css`: Custom CSS styling for the app.
- `brain_tumor_classifier.h5`: Trained model file.
- `requirements.txt`: Python dependencies.
- `README.md`: Project documentation.
- `images/`: Directory for images used in the README or app.

## **Model Details**

The deep learning model is based on DenseNet201, a pre-trained convolutional neural network from Keras Applications. The model has been fine-tuned to classify brain MRI images into the following categories:

- Glioma
- Meningioma
- Pituitary Tumor
- Healthy

### **Training Dataset**

The model was trained on a dataset containing MRI images labeled with the four categories. Data augmentation and transfer learning techniques were used to enhance performance.

## **Dependencies**

- Python 3.7 - 3.10
- Streamlit
- TensorFlow
- Keras
- Pillow
- NumPy
- Pandas
- Altair

Install all dependencies using:

```bash
pip install -r requirements.txt
```

## **Sample `requirements.txt`**

```plaintext
streamlit
tensorflow
pillow
numpy
pandas
altair
```

## **Contributing**

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

### **Steps to Contribute**

1. Fork the repository.
2. Create a new branch:

```bash
git checkout -b feature/YourFeature
```

3. Commit your changes:

```bash
git commit -m 'Add Your Feature'
```

4. Push to the branch:

```bash
git push origin feature/YourFeature
```

5. Open a pull request.

## **License**

This project is licensed under the MIT License. See the LICENSE file for details.

## **Disclaimer**

### **Medical Disclaimer**: This app is intended for educational purposes only and should not be used for medical diagnostics. Always consult a healthcare professional for medical advice.

### **Data Privacy**: The app does not store any uploaded images or personal data. All processing is done in memory.

## **Acknowledgements**

- **Dataset**: Brain MRI Images for Brain Tumor Detection
- **Model Architecture**: DenseNet201
- **Inspiration**: The need for accessible tools to aid in medical imaging analysis.

## **Contact**

For any questions or suggestions, feel free to reach out:

- **Name**: Atef Hassan
- **Email**: atefabdo26399@gmail.com
- **GitHub**: [Atef Hassan](https://github.com/atef)
- **LinkedIn**: [Atef Hassan](https://linkedin.com/in/atef)

(Replace the GitHub and LinkedIn links with your actual profiles if available.)

## **Future Work**

- Expand Classification Categories: Include more types of brain tumors and conditions.
- Improve Model Accuracy: Continuously update the model with more data and better techniques.
- Deploy on Cloud Platform: Make the app accessible online for wider use.
- Add Multi-Language Support: Cater to non-English speaking users.

## **Screenshots**

- Home Page
- Prediction Results

(Add actual screenshots of your app in the `images/` directory and update the image paths accordingly.)

## **How It Works**

1. **Upload**: Users upload an MRI image through the app interface.
2. **Preprocessing**: The image is preprocessed to match the input requirements of the model (resizing, normalization).
3. **Prediction**: The pre-trained model processes the image and outputs probabilities for each class.
4. **Results Display**: The app displays the predicted class, confidence scores, and additional information.
5. **Report Generation**: Users can download a report containing the results.

## **Technical Details**

- **Framework**: The app is built using Streamlit, allowing for rapid development of data apps.
- **Model Loading**: Utilizes `@st.cache_resource` to load the model efficiently and prevent unnecessary reloads.
- **Visualization**: Employs Altair for interactive and responsive charts.
- **Error Handling**: Implements `try-except` blocks to handle invalid inputs gracefully.

## **Security Considerations**

- **File Handling**: Uploaded images are processed in memory and not saved to disk.
- **User Data**: No personal data is collected or stored.
- **Dependencies**: All libraries used are widely adopted and regularly updated to patch vulnerabilities.

## **Frequently Asked Questions (FAQ)**

1. **Can I use this app for medical diagnosis?**

   No. This app is intended for educational purposes only. It should not be used as a substitute for professional medical advice, diagnosis, or treatment.

2. **What types of images are supported?**

   The app supports MRI images in JPG and PNG formats.

3. **How accurate is the model?**

   The model's accuracy depends on the quality and diversity of the training data. While it performs well on the dataset it was trained on, it may not generalize perfectly to all real-world data.

4. **Is my uploaded data secure?**

   Yes. The app does not store any uploaded images or personal data. All processing is done locally on your machine when running the app.

## **Feedback**

If you encounter any issues or have suggestions for improvements, please open an issue on the GitHub repository.

## **Acknowledgments**

- **Contributors**: Thanks to everyone who has contributed to this project.
- **Open Source Community**: This project leverages several open-source libraries and frameworks.

---

### **Clone the Repository**

```bash
git clone https://github.com/yourusername/CerebroClassif.git
cd CerebroClassif
```
