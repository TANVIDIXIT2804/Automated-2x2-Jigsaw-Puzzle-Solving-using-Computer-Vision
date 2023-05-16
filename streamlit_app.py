import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import os
from io import StringIO
import warnings

# Import your jigsaw solver model here
model = tf.keras.models.load_model('model_new_1.h5')
model.summary()

def tiles_images(img, pieces):
    # print(img)
    img_width = img_height = img.shape[1] // pieces
    tiles_img = [img[x:x+img_height,y:y+img_width]
                 for x in range(0,img.shape[0], img_height)
                 for y in range(0,img.shape[1],img_width)
                ]
    
    new_tiles = []
    
    for img in tiles_img:
        if img.shape[0] == img_width and  img.shape[1] == img_height:
            new_tiles.append(img)

    return new_tiles

def group_image(images, shuffle_arr, pieces=2):
    img = []
    
    new_array = [shuffle_arr[i] for i in shuffle_arr]
    new_order =  [images[i] for i in new_array]

    i1= np.concatenate((images[shuffle_arr[0]],images[shuffle_arr[1]]),axis =1)
    i2 = np.concatenate((images[shuffle_arr[2]],images[shuffle_arr[3]]),axis=1)
    
    
    img = np.concatenate((i1,i2), axis=0)
    return img

# Define the path to the Gravity Falls folder
gravity_falls_path = os.path.join(os.getcwd(), 'Gravity Falls', 'puzzle_2x2', 'test')

# Get a list of all image files in the Gravity Falls folder
image_files = [f for f in os.listdir(gravity_falls_path) if f.endswith('.jpg')]

# Construct a list of full file paths for the images
test_image_paths = [os.path.join(gravity_falls_path, f) for f in image_files]



# Define the CSS styles
css = """
[data-testid="stSidebar"] {
    background-color: #5F9EA0;
}
"""

# Apply the CSS styles using st.markdown
st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

# Add some elements to the sidebar
st.sidebar.header("Image Selector")
# Define a slider to select a test image
selected_image_index = st.sidebar.slider(
    'Select an image',
    min_value=0,
    max_value=len(test_image_paths)-1,
    value=0,
    step=1
)

# Load the selected test image
test_image = Image.open(test_image_paths[selected_image_index])
# Image.show(test_image)
# Convert the test image to a numpy array
test_image = np.array(test_image).astype('float16')/255 -0.5
# print(test_image)

inp = np.expand_dims(tiles_images(test_image, pieces=2), axis=0)

y_pred = model.predict(inp)[0]
y_pred = np.argmax(y_pred, axis=1)
print(y_pred)

tiles = tiles_images(test_image, pieces=2)
predit_img = group_image(tiles, y_pred, pieces=2)

# Define the CSS styles
# page_bg_img = '''
# <style>
# body {
#     background-image: url('https://img.freepik.com/free-photo/white-puzzle-pieces-blue-background_641386-665.jpg');
#     background-size: cover;
#     color: red;
# }
# </style>
# '''

# st.markdown(page_bg_img, unsafe_allow_html=True)

st.markdown('<style>body{background-color: Blue;}</style>',unsafe_allow_html=True)


st.title("2x2 JIGSAW")
st.header("Puzzle Generator and Solver")

col1, col2 = st.columns(2)

# Display the first image in the first column
with col1:
    # Display the original image, and the solved puzzle
    st.image(test_image,clamp = True, caption='Test Puzzle')

# Display the second image in the second column
with col2:
    #st.image(np.concatenate(tiles, axis=1), caption='Shuffled Tiles')
    st.image(predit_img,clamp = True, caption='Solved Puzzle')


# Define custom CSS styles for the model summary
st.write(
    f"""
    <style>
        .model-summary {{
            font-family: monospace;
            white-space: pre;
            font-size: 12px;
        }}
    </style>
    """
)

# Create an expander
with st.expander("Model Summary", expanded=False):
    # Use StringIO to capture the model summary as a string
    summary_string = StringIO()
    model.summary(print_fn=lambda x: summary_string.write(x + "\n"))
    summary = summary_string.getvalue()

    # Display the model summary using st.write and apply the custom CSS styles
    st.write(f"<pre class='model-summary'>{summary}</pre>", unsafe_allow_html=True)