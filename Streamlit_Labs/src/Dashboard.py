import json
import requests
import streamlit as st
from pathlib import Path
from streamlit.logger import get_logger

# If you start the fast api server on a different port
# make sure to change the port below
FASTAPI_BACKEND_ENDPOINT = "http://localhost:8000"

# Make sure you have iris_model.pkl file in FastAPI_Labs/src folder.
# If it's missing run train.py in FastAPI_Labs/src folder
# If your FastAPI_Labs folder name is different, update accordingly in the following path
FASTAPI_WINE_MODEL_LOCATION = Path(__file__).resolve().parents[2] / 'FastAPI_Labs' / 'model' / 'wine_model.pkl'

# streamlit logger
LOGGER = get_logger(__name__)

def run():
    # Set the main dashboard page browser tab title and icon
    st.set_page_config(
        page_title="Wine Prediction",
        page_icon="🍷",
    )

    # Build the sidebar first
    # This sidebar context gives access to work on elements in the side panel
    with st.sidebar:
        # Check the status of backend
        try:
            # Make sure fast api is running. Check the lab for guidance on getting
            # the server up and running
            backend_request = requests.get(FASTAPI_BACKEND_ENDPOINT)
            # If backend returns successful connection (status code: 200)
            if backend_request.status_code == 200:
                # This creates a green box with message
                st.success("Backend online ✅")
            else:
                # This creates a yellow bow with message
                st.warning("Problem connecting 😭")
        except requests.ConnectionError as ce:
            LOGGER.error(ce)
            LOGGER.error("Backend offline 😱")
            # Show backend offline message
            st.error("Backend offline 😱")

        st.info("Configure parameters")
        # Set the values
        proline = st.number_input("Proline", min_value=278.0, max_value=1680.0, value=278.0, step=1.0, format="%f")
        flavanoids = st.number_input("Flavanoids", min_value=0.33, max_value=5.08, value=0.33, step=0.01, format="%f")
        color_intensity = st.number_input("Color Intensity", min_value=1.28, max_value=13.0, value=1.28, step=0.01, format="%f")
        # sepal_width = st.slider("Sepal Width",2.0, 4.4, 2.0, 0.1, help="Sepal width in centimeter (cm)", format="%f")
        # petal_length = st.slider("Petal Length",1.0, 6.9, 1.0, 0.1, help="Petal length in centimeter (cm)", format="%f")
        # petal_width = st.slider("Petal Width",0.1, 2.5, 0.1, 0.1, help="Petal width in centimeter (cm)", format="%f")
        
        # Take JSON file as input
        test_input_file = st.file_uploader('Upload test prediction file',type=['json'])

        # Check if client has provided input test file
        if test_input_file:
            # Quick preview functionality for JSON input file
            st.write('Preview file')
            test_input_data = json.load(test_input_file)
            st.json(test_input_data)
            # Session is necessary, because the sidebar context acts within a 
            # scope, so to access information outside the scope
            # we need to save the information into a session variable
            st.session_state["IS_JSON_FILE_AVAILABLE"] = True
        else:
            # If user adds file, then performs prediction and then removes
            # file, the session var should revert back since file 
            # is not available
            st.session_state["IS_JSON_FILE_AVAILABLE"] = False
            
        # Predict button
        predict_button = st.button('Predict')

    # Dashboard body
    # Heading for the dashboard
    st.write("# Wine Prediction! 🍷")
    # If predict button is pressed
    if predict_button:
        result_container = st.empty()

        if FASTAPI_WINE_MODEL_LOCATION.is_file():
        # Choose input: JSON file if available, otherwise sliders/number inputs
            if st.session_state.get("IS_JSON_FILE_AVAILABLE", False):
                client_input = test_input_data['input_test']  # JSON input
            else:
                client_input = {  # Slider/number input
                        "flavanoids": flavanoids,
                        "color_intensity": color_intensity,
                        "proline": proline
                        }

        # Send to FastAPI
            try:
                with st.spinner("Predicting..."):
                    predict_wine_response = requests.post(
                        f"{FASTAPI_BACKEND_ENDPOINT}/predict",
                        json=client_input
                    )
                if predict_wine_response.status_code == 200:
                    wine_content = predict_wine_response.json()
                    start_sentence = "The wine predicted is: "
                    if wine_content["response"] == 0:
                        result_container.success(f"{start_sentence} Barolo")
                    elif wine_content["response"] == 1:
                        result_container.success(f"{start_sentence} Grignolino")
                    elif wine_content["response"] == 2:
                        result_container.success(f"{start_sentence} Barbera")
                    else:
                        result_container.error("Unexpected prediction result")
                else:
                    result_container.error(f"Server returned status {predict_wine_response.status_code}")
            except Exception as e:
                result_container.error(f"Prediction failed: {e}")

    else:
        st.error("Model file not found. Run train.py first.")


if __name__ == "__main__":
    run()