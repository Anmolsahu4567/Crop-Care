import streamlit as st
import tensorflow as tf
import numpy as np

#Tensorflow Model Prediction
def model_prediction(test_image):
    try:
        model = tf.keras.models.load_model('trained_model.keras')
    except (OSError, IOError):
        st.error(
            "⚠️ Model file 'trained_model.keras' was not found. "
            "Please download it (see the README) and place it in the project's root folder."
        )
        st.stop()

    try:
        image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    except Exception:
        st.error("⚠️ Could not read the uploaded image. Please try a different file.")
        st.stop()

    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to a batch
    prediction = model.predict(input_arr)
    result_index = np.argmax(prediction)
    return result_index

#Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page",["Home","About","Disease Recognition"])

#Home Page
# Home Page
if(app_mode=="Home"):

    # Big title area
    st.markdown("""
    <div style="
        background: linear-gradient(120deg, #14532d, #022c22);
        padding: 20px 25px;
        border-radius: 16px;
        margin-bottom: 20px;">
        <h1 style="color:#ecfdf5; margin-bottom:5px;">🌿 CROP CARE</h1>
        <p style="color:#bbf7d0; font-size:16px; margin-top:0;">
            AI-powered Plant Disease Recognition to help farmers and students detect crop diseases early.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1.2, 1.8])

    with col1:
        st.subheader("Welcome to the Plant Disease Recognition System! 🌾")
        st.markdown("""
        Our mission is to help in identifying plant diseases efficiently.  
        Upload an image of a plant leaf, and our system will analyze it to detect any signs of disease.

        ### 🌟 Why This App?
        - ✅ **Accurate**: Uses a trained deep learning model for disease detection  
        - ⚡ **Fast**: Get results in just a few seconds  
        - 🧑‍🌾 **Farmer & Student Friendly**: Simple, clean and easy to understand  

        ### 🚀 How It Works
        1. Go to the **Disease Recognition** page from the sidebar  
        2. Upload a clear image of a **single plant leaf**  
        3. Click on **Predict** and see the result  

        ### 🌱 Small Note
        This tool is for **preliminary guidance**. For final treatment and pesticide decisions,
        always consult an agriculture expert.
        """)

    with col2:
        image_path = "home_page.jpeg"
        st.image(image_path, use_container_width=True)

        st.markdown("""
        ### 📊 Quick Overview
        - 🔍 Detects **38+** types of plant diseases  
        - 🌿 Supports multiple crops like **Apple, Grape, Tomato, Potato, Corn** and more  
        - 🧠 Built using **TensorFlow & Keras**

        ### 💡 Tip for Best Results
        - Use a clear close-up of the leaf  
        - Avoid blur / very dark images  
        - Try to keep only one leaf in the frame  
        """)

    st.markdown("---")

    # Three feature cards
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div style="
            background-color:#022c22;
            padding:15px;
            border-radius:12px;
            text-align:center;">
            <h4 style="color:#a7f3d0;">🤖 AI Powered</h4>
            <p style="color:#e5e7eb; font-size:13px;">
                Deep learning model trained on thousands of leaf images.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div style="
            background-color:#111827;
            padding:15px;
            border-radius:12px;
            text-align:center;">
            <h4 style="color:#facc15;">🧑‍🌾 For Farmers & Students</h4>
            <p style="color:#e5e7eb; font-size:13px;">
                Simple interface designed for easy understanding and learning.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div style="
            background-color:#1f2933;
            padding:15px;
            border-radius:12px;
            text-align:center;">
            <h4 style="color:#93c5fd;">📚 Learn & Explore</h4>
            <p style="color:#e5e7eb; font-size:13px;">
                Use it as a learning tool to understand crop diseases better.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    ---
    ### ℹ️ About :
    - We aim to combine **AI + Agriculture** to make crop care smarter and more accessible
    - Designed to support **farmers, agri students, and engineering projects**
    - **Future Scope:** Multilingual support, additional crops & diseases, and a mobile app version
    """)

#About Page
elif(app_mode=="About"):
    st.header("ℹ️ About Crop Care")

    st.markdown("""
    Crop Care is an AI-based Plant Disease Recognition system designed to help
    farmers, agriculture students, and researchers quickly identify crop diseases
    using leaf images.
    """)

    col1, col2 = st.columns(2)

    # Left column: Dataset details
    with col1:
        st.subheader("📂 Dataset Information")
        st.markdown("""
        - Total images: **~87,000+ RGB leaf images**  
        - Classes: **38** (healthy + various diseases)  
        - Crops include: **Apple, Grape, Tomato, Potato, Corn**, and more  
        - Images contain both **healthy and diseased** leaves  

        #### Dataset Split
        1. **Train:** 70,295 images  
        2. **Validation:** 17,572 images  
        3. **Test:** 33 images  

        The dataset is recreated using **offline augmentation** from the
        original dataset while preserving directory structure.
        """)

    # Right column: Model & tech details
    with col2:
        st.subheader("🧠 Model & Tech Stack")
        st.markdown("""
        - Framework: **TensorFlow / Keras**  
        - Model Type: **Convolutional Neural Network (CNN)**  
        - Input Image Size: **128 × 128** pixels  
        - Task: **Multi-class classification (38 classes)**  

        #### Technology Used
        - **Python** for development  
        - **Streamlit** for the web interface  
        - **NumPy, TensorFlow, Keras** for deep learning  
        - Image preprocessing using **Keras Preprocessing APIs**
        """)

    st.markdown("---")

    st.subheader("👨‍💻 About the Developer")
    st.markdown("""
    - Developed by **Anmol Sahu**  
    - Student of **B.Tech – Computer Science (AI & ML)**  
    - **Gyan Ganga Institute of Technology and Sciences, Jabalpur**  

    We aim to combine **Artificial Intelligence + Agriculture** to make crop care
    smarter, faster, and more accessible to everyone.

    #### 🌱 Future Scope
    - Support for more crops and disease classes  
    - **Multilingual interface** (English + regional languages)  
    - Mobile app version for on-field use  
    - More detailed **treatment suggestions** and expert guidance integration  
    """)
    
#Prediction Page
# ==========================
# Prediction Page (Enhanced UI)
# ==========================
elif(app_mode=="Disease Recognition"):
    st.header("🌿 Crop Disease Recognition & Guidance")

    st.markdown("""
    <div style="
        background-color:#1f2933;
        padding:18px;
        border-radius:12px;
        margin-bottom:20px;">
        <h4 style="color:#7dd3fc;">Instructions</h4>
        <ul style="color:#e5e7eb;">
            <li>Upload a clear image of a <b>single leaf</b></li>
            <li>Use good lighting (no dark shadows)</li>
            <li>Avoid blurry or distant photos</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    test_image = st.file_uploader(
        "📤 Upload Leaf Image",
        type=["jpg", "jpeg", "png"]
    )

    # Image Preview
    if test_image is not None:
        st.subheader("📸 Image Preview")
        st.image(test_image,use_container_width=True)
    else:
        st.info("Please upload a leaf image to continue.")

    # Action Buttons
    col1, col2 = st.columns(2)

    with col1:
        show_img = st.button("👁️ Show Image")
    with col2:
        predict_btn = st.button("🔍 Predict Disease")

    if show_img:
        if test_image is None:
            st.warning("Upload an image first.")
        else:
            st.image(test_image, use_container_width=True)

    # Predict Button Logic
    if predict_btn:
        if test_image is None:
            st.error("❌ Please upload an image before predicting.")
        else:
            with st.spinner("Analyzing the leaf image..."):
                st.write("### 🔎 Our Prediction")

                result_index = model_prediction(test_image)

                class_name = [
                    'Apple___Apple_scab',
                    'Apple___Black_rot',
                    'Apple___Cedar_apple_rust',
                    'Apple___healthy',
                    'Blueberry___healthy',
                    'Cherry_(including_sour)___Powdery_mildew',
                    'Cherry_(including_sour)___healthy',
                    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                    'Corn_(maize)___Common_rust_',
                    'Corn_(maize)___Northern_Leaf_Blight',
                    'Corn_(maize)___healthy',
                    'Grape___Black_rot',
                    'Grape___Esca_(Black_Measles)',
                    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                    'Grape___healthy',
                    'Orange___Haunglongbing_(Citrus_greening)',
                    'Peach___Bacterial_spot',
                    'Peach___healthy',
                    'Pepper,_bell___Bacterial_spot',
                    'Pepper,_bell___healthy',
                    'Potato___Early_blight',
                    'Potato___Late_blight',
                    'Potato___healthy',
                    'Raspberry___healthy',
                    'Soybean___healthy',
                    'Squash___Powdery_mildew',
                    'Strawberry___Leaf_scorch',
                    'Strawberry___healthy',
                    'Tomato___Bacterial_spot',
                    'Tomato___Early_blight',
                    'Tomato___Late_blight',
                    'Tomato___Leaf_Mold',
                    'Tomato___Septoria_leaf_spot',
                    'Tomato___Spider_mites Two-spotted_spider_mite',
                    'Tomato___Target_Spot',
                    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
                    'Tomato___Tomato_mosaic_virus',
                    'Tomato___healthy'
                ]

                predicted_class = class_name[result_index]

                # Result Card
                st.markdown(f"""
                <div style="
                    background: linear-gradient(135deg, #064e3b, #022c22);
                    padding:25px;
                    border-radius:16px;
                    text-align:center;
                    margin-top:20px;">
                    <h2 style="color:#a7f3d0;">✅ Prediction Result</h2>
                    <h3 style="color:#ecfeff;">{predicted_class}</h3>
                    <p style="color:#bbf7d0; font-size:15px;">
                        The model has identified this disease based on the uploaded image.
                    </p>
                </div>
                """, unsafe_allow_html=True)

                # Additional tips
                with st.expander("🌱 What should you do next?"):
                    st.markdown("""
                    - Compare the result with visible symptoms on your plant.
                    - Remove infected leaves if possible.
                    - Avoid over-watering.
                    - Consult an agriculture expert for chemical treatment.
                    """)

                with st.expander("⚠️ Important Disclaimer"):
                    st.write("""
                    This prediction is generated by a Machine Learning model and is for
                    **educational and preliminary guidance only**.  
                    For accurate treatment and pesticide usage, always consult a certified
                    agricultural officer or plant pathologist.
                    """)

                st.success("✅ Model prediction completed successfully!")
