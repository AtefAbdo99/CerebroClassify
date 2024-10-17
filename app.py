<<<<<<< HEAD
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import pandas as pd
import base64
import altair as alt
import os

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Set page configuration
st.set_page_config(
    page_title="Brain Tumor Classifier",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Hide Streamlit default style elements
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

# Load custom CSS
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

load_css("style.css")

# Load your trained model
@st.cache_resource
def load_model_cached():
    model = load_model('brain_tumor_classifier.h5')
    return model

model = load_model_cached()

# Define the class names and explanations
class_info = {
    'Glioma': 'A type of tumor that occurs in the brain and spinal cord originating from glial cells.',
    'Healthy': 'No tumor detected. The brain appears to be healthy.',
    'Meningioma': 'A tumor that arises from the meninges, the membranes that surround your brain and spinal cord.',
    'Pituitary': 'A tumor that develops in the pituitary gland at the base of the brain.'
}

class_names = list(class_info.keys())

# Preprocess image
def preprocess_image(image):
    IMG_HEIGHT, IMG_WIDTH = 224, 224
    image = image.convert('RGB')
    image = ImageOps.fit(image, (IMG_WIDTH, IMG_HEIGHT), Image.Resampling.LANCZOS)
    image_array = np.array(image) / 255.0  # Normalize pixel values
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

# Background image function (optional)
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image:
        encoded_string = base64.b64encode(image.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    </style>
    """,
    unsafe_allow_html=True
    )

# Uncomment the next line to add a background image
# add_bg_from_local('background.png')

# Main Application
def main():
    # App title and description
    st.title("🧠 Brain Tumor Classification")
    st.markdown("""
    Welcome to the **Brain Tumor Classification App**. Upload an MRI scan image, and the app will predict the type of brain tumor.

    **Instructions:**
    - Upload an MRI image in **JPG** or **PNG** format.
    - Click on **Predict** to see the results.
    """)

    # Sidebar information
    st.sidebar.title("About")
    st.sidebar.info("""
    This app uses a **Deep Learning model** (DenseNet201) to classify brain MRI scans into four categories:

    - **Glioma**
    - **Healthy**
    - **Meningioma**
    - **Pituitary**

    *Created for educational purposes.*
    """)

    # Add an expander for explanations
    with st.expander("ℹ️ What do these diagnoses mean?"):
        st.markdown("""
        **Glioma**: A tumor arising from the glial cells in the brain or spine.

        **Healthy**: No signs of tumor detected.

        **Meningioma**: A tumor that forms on membranes covering the brain and spinal cord.

        **Pituitary**: A tumor in the pituitary gland, affecting hormone production.
        """)

    # File uploader
    uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"], accept_multiple_files=False)

    if uploaded_file is not None:
        try:
            # Read the image file
            image = Image.open(uploaded_file)
            # Display the uploaded image
            st.image(image, caption='Uploaded MRI Image', use_column_width=True)
        except Exception as e:
            st.error('Invalid image file. Please upload a valid MRI image.')
            st.stop()

        # Add a predict button
        if st.button('Predict'):
            with st.spinner('Analyzing...'):
                # Preprocess the image
                image_array = preprocess_image(image)
                # Make predictions
                preds = model.predict(image_array)
                pred_class = class_names[np.argmax(preds[0])]
                confidence = np.max(preds[0]) * 100

                # Display prediction
                st.markdown("## Results")
                st.success(f'**Prediction:** {pred_class}')
                st.info(f'**Confidence:** {confidence:.2f}%')

                # Explanation for the predicted class
                st.markdown(f"**About {pred_class}:** {class_info[pred_class]}")

                # Link to more information
                st.markdown(f"**Learn more about {pred_class} tumors [here](https://en.wikipedia.org/wiki/Neuroimaging).**")

                # Display confidence scores as a bar chart
                scores = {class_names[i]: float(preds[0][i]) * 100 for i in range(len(class_names))}
                scores_df = pd.DataFrame(scores.items(), columns=['Diagnosis', 'Confidence'])
                scores_df = scores_df.sort_values(by='Confidence', ascending=True)

                st.markdown("### Confidence Scores")
                bar_chart = alt.Chart(scores_df).mark_bar().encode(
                    x=alt.X('Confidence:Q', axis=alt.Axis(format='.2f'), title='Confidence (%)'),
                    y=alt.Y('Diagnosis:N', sort='-x', title='Diagnosis'),
                    color=alt.Color('Diagnosis:N', legend=None),
                    tooltip=['Diagnosis', 'Confidence']
                ).properties(height=300)
                st.altair_chart(bar_chart, use_container_width=True)

                # Provide an option to download the results
                def generate_report(pred_class, confidence, scores_df):
                    report = f"""
Brain Tumor Classification Report

Prediction: {pred_class}
Confidence: {confidence:.2f}%

Confidence Scores:
{scores_df.to_string(index=False)}

Note: This result is for informational purposes only and should not be considered a medical diagnosis. Please consult a healthcare professional for medical advice.
"""
                    return report

                report = generate_report(pred_class, confidence, scores_df)
                b64 = base64.b64encode(report.encode()).decode()
                href = f'<a href="data:file/txt;base64,{b64}" download="report.txt">📥 Download Report</a>'
                st.markdown(href, unsafe_allow_html=True)

                # Remove the uploaded file from memory
                uploaded_file.close()
    else:
        st.info('Please upload an MRI image to get started.')

    # Footer
    st.markdown("""
    <hr style="border:1px solid gray">

    <div style="text-align: center;">
        Developed with ❤️ By Atef Hassan for educational purposes | © 2024
    </div>
    """, unsafe_allow_html=True)

if __name__ == '__main__':
    main()
=======
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import pandas as pd
import base64
import altair as alt
import os

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Set page configuration
st.set_page_config(
    page_title="Brain Tumor Classifier",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Hide Streamlit default style elements
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

# Load custom CSS
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

load_css("style.css")

# Load your trained model
@st.cache_resource
def load_model_cached():
    model = load_model('brain_tumor_classifier.h5')
    return model

model = load_model_cached()

# Define the class names and explanations
class_info = {
    'Glioma': 'A type of tumor that occurs in the brain and spinal cord originating from glial cells.',
    'Healthy': 'No tumor detected. The brain appears to be healthy.',
    'Meningioma': 'A tumor that arises from the meninges, the membranes that surround your brain and spinal cord.',
    'Pituitary': 'A tumor that develops in the pituitary gland at the base of the brain.'
}

class_names = list(class_info.keys())

# Preprocess image
def preprocess_image(image):
    IMG_HEIGHT, IMG_WIDTH = 224, 224
    image = image.convert('RGB')
    image = ImageOps.fit(image, (IMG_WIDTH, IMG_HEIGHT), Image.Resampling.LANCZOS)
    image_array = np.array(image) / 255.0  # Normalize pixel values
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

# Background image function (optional)
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image:
        encoded_string = base64.b64encode(image.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    </style>
    """,
    unsafe_allow_html=True
    )

# Uncomment the next line to add a background image
# add_bg_from_local('background.png')

# Main Application
def main():
    # App title and description
    st.title("🧠 Brain Tumor Classification")
    st.markdown("""
    Welcome to the **Brain Tumor Classification App**. Upload an MRI scan image, and the app will predict the type of brain tumor.

    **Instructions:**
    - Upload an MRI image in **JPG** or **PNG** format.
    - Click on **Predict** to see the results.
    """)

    # Sidebar information
    st.sidebar.title("About")
    st.sidebar.info("""
    This app uses a **Deep Learning model** (DenseNet201) to classify brain MRI scans into four categories:

    - **Glioma**
    - **Healthy**
    - **Meningioma**
    - **Pituitary**

    *Created for educational purposes.*
    """)

    # Add an expander for explanations
    with st.expander("ℹ️ What do these diagnoses mean?"):
        st.markdown("""
        **Glioma**: A tumor arising from the glial cells in the brain or spine.

        **Healthy**: No signs of tumor detected.

        **Meningioma**: A tumor that forms on membranes covering the brain and spinal cord.

        **Pituitary**: A tumor in the pituitary gland, affecting hormone production.
        """)

    # File uploader
    uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"], accept_multiple_files=False)

    if uploaded_file is not None:
        try:
            # Read the image file
            image = Image.open(uploaded_file)
            # Display the uploaded image
            st.image(image, caption='Uploaded MRI Image', use_column_width=True)
        except Exception as e:
            st.error('Invalid image file. Please upload a valid MRI image.')
            st.stop()

        # Add a predict button
        if st.button('Predict'):
            with st.spinner('Analyzing...'):
                # Preprocess the image
                image_array = preprocess_image(image)
                # Make predictions
                preds = model.predict(image_array)
                pred_class = class_names[np.argmax(preds[0])]
                confidence = np.max(preds[0]) * 100

                # Display prediction
                st.markdown("## Results")
                st.success(f'**Prediction:** {pred_class}')
                st.info(f'**Confidence:** {confidence:.2f}%')

                # Explanation for the predicted class
                st.markdown(f"**About {pred_class}:** {class_info[pred_class]}")

                # Link to more information
                st.markdown(f"**Learn more about {pred_class} tumors [here](https://en.wikipedia.org/wiki/Neuroimaging).**")

                # Display confidence scores as a bar chart
                scores = {class_names[i]: float(preds[0][i]) * 100 for i in range(len(class_names))}
                scores_df = pd.DataFrame(scores.items(), columns=['Diagnosis', 'Confidence'])
                scores_df = scores_df.sort_values(by='Confidence', ascending=True)

                st.markdown("### Confidence Scores")
                bar_chart = alt.Chart(scores_df).mark_bar().encode(
                    x=alt.X('Confidence:Q', axis=alt.Axis(format='.2f'), title='Confidence (%)'),
                    y=alt.Y('Diagnosis:N', sort='-x', title='Diagnosis'),
                    color=alt.Color('Diagnosis:N', legend=None),
                    tooltip=['Diagnosis', 'Confidence']
                ).properties(height=300)
                st.altair_chart(bar_chart, use_container_width=True)

                # Provide an option to download the results
                def generate_report(pred_class, confidence, scores_df):
                    report = f"""
Brain Tumor Classification Report

Prediction: {pred_class}
Confidence: {confidence:.2f}%

Confidence Scores:
{scores_df.to_string(index=False)}

Note: This result is for informational purposes only and should not be considered a medical diagnosis. Please consult a healthcare professional for medical advice.
"""
                    return report

                report = generate_report(pred_class, confidence, scores_df)
                b64 = base64.b64encode(report.encode()).decode()
                href = f'<a href="data:file/txt;base64,{b64}" download="report.txt">📥 Download Report</a>'
                st.markdown(href, unsafe_allow_html=True)

                # Remove the uploaded file from memory
                uploaded_file.close()
    else:
        st.info('Please upload an MRI image to get started.')

    # Footer
    st.markdown("""
    <hr style="border:1px solid gray">

    <div style="text-align: center;">
        Developed with ❤️ By Atef Hassan for educational purposes | © 2024
    </div>
    """, unsafe_allow_html=True)

if __name__ == '__main__':
    main()
>>>>>>> 64f94fea8c0a5f4a0dea10f72458c0aa56733b7e
