##### version be4 20dec20
import detectron2

# import some common libraries
import os
import numpy as np
import cv2
import random
import streamlit as st
import tensorflow as tf
import pandas as pd 
from pathlib import Path
from PIL import Image

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg

@st.cache
def category_model():

    cfg = get_cfg()

    cfg.MODEL.DEVICE='cpu'

    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))

    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 13

    cfg.MODEL.WEIGHTS = './model_final_202012161200.pth'

    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.65   # set the testing threshold for this model

    predictor = DefaultPredictor(cfg)
    
    return predictor

@st.cache
def get_image(buffer):
    # detectron model is trained using channel BGR
    # convert the input image from RGB to BGR
    im = np.asarray(Image.open(buffer))[:,:,::-1]
    return im

@st.cache
def color_model():
    f = Path(r'./color_model_structure_161220.json')
    model_structure = f.read_text()
    model = tf.keras.models.model_from_json(model_structure)
    model.load_weights(r'./color_model_weight_161220.h5')
    return model

@st.cache
def attribute_model():
    f = Path(r'./att_model_161220.json')
    model_structure = f.read_text()
    model = tf.keras.models.model_from_json(model_structure)
    model.load_weights(r'./att_model_161220.h5')
    return model

@st.cache
def preprocess(img, mode):
    if mode == 'c':
        target_size = (227,227) #input image size of the color model
    elif mode == 'a':
        target_size = (256,256) #input image size of the attribute model
    else: 
        return 'insert mode: either color or att'
    resized = img.resize(target_size, Image.NEAREST)
    processed = np.array(resized, dtype='float32').reshape(1,*target_size,3)
    processed /= 255
    return processed

@st.cache
def load_catalog():
    df = pd.read_pickle('./final_dataframe2.pkl')
    return df

def scoring(rois, outputs, c_model, a_model):
    full_score = []
    if len(rois) > 0:
        for i in range(len(rois)):
            #initiatize scores for each item in the input image
            item_scores = [0] *13

            #set score for category
            cat_index = int(outputs['instances'].pred_classes[i])
            item_scores[cat_index] = 1

            #set scores for colors
            c_scores = c_model.predict(preprocess(rois[i], 'c'))[0] 
            item_scores.extend(c_scores)

            #set scores for attributes
            a_scores = a_model.predict(preprocess(rois[i], 'a'))[0] 
            item_scores.extend(a_scores)

            full_score.append(item_scores)
            
        return full_score
    else:
        st.write('no item detected.')
        
def recommend(input_score, catalog_scores):
    indices = []
    for i in input_score:
        dot_product_result = pd.DataFrame(i).T.dot(catalog_scores.T).T 

        index_top = np.random.permutation(dot_product_result.nlargest(5,0).index) #
        
        indices.append(index_top[0:5])
    return indices
        
def main():
    st.title('EcoWear')
    
    """
    New to sustainable fashion? No problem, we are here to help!
    Just upload an image of a clothing item you like, be it t-shirt, shirt, jacket, pants, etc.
    and we will find you the closest alternatives offered by verified ethical fashion sites.
    """

    # load the weights and architectures of the 3 models
    cat_model = category_model()
    c_model = color_model()
    a_model = attribute_model()
    
    # load the catalog dataframe for recommendation
    catalog = load_catalog()
    catalog_scores = catalog.loc[:, 3:33].copy()
    catalog_scores.columns = range(0,31)

    with st.spinner('*LOADING*...be patient, good things take time :sunglasses:'):  
        input_img_buffer = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])
    
        if input_img_buffer == None:
            st.warning('No file uploaded')
            
        elif input_img_buffer:
            img = get_image(input_img_buffer) # The image is turned to BGR channels

            with st.beta_container():
                cols = st.beta_columns([250]*2)
                with cols[0]:
                    st.write('You uploaded:')
                    st.image(img[:,:,::-1], width=250)
                with cols[1]:
                    placeholder = st.empty()
                    placeholder2 = st.empty()

            outputs = cat_model(img)
            items_detected = len(outputs['instances'].pred_classes)

            if items_detected > 0:
                # store the fashion item images segmented by detectron2 model
                rois = [Image.fromarray(img[:, :, ::-1]).crop(i.__array__()) for i in outputs['instances'].get_fields()['pred_boxes']]

                scores = scoring(rois, outputs, c_model, a_model) # turning each of the item images into 31-feature array
                if items_detected > 1:
                    option = placeholder.selectbox(f'{items_detected} item(s) detected, which one do you want to look into?', list(range(1,items_detected+1)))
                    
                elif items_detected == 1:
                    option = 1    
                
                base_dir = './image/'

                with st.beta_container():
                    recommendations = recommend(scores, catalog_scores)
                    recommend_details = catalog.loc[recommendations[option-1]].copy()

                    cat_name = ['short_sleeved_shirt', 'long_sleeved_shirt', 'short_sleeved_outwear', 'long_sleeved_outwear',
                                'vest', 'sling', 'shorts', 'trousers', 'skirt', 'short_sleeved_dress',
                                'long_sleeved_dress', 'vest_dress', 'sling_dress']
                    color_map = {0: 'Black', 1: 'Blue', 2: 'Green', 3: 'Grey', 4: 'Orange', 5: 'Pink',
                                6: 'Purple', 7: 'Red', 8: 'White', 9: 'Yellow'}
                    att_map = {0: 'lace', 1: 'graphic', 2: 'floral', 3: 'stripes', 4: 'pattern', 5: 'pocket', 
                                6: 'print', 7: 'v-neckline'}
                    cat = cat_name[np.argmax(scores[option-1][0:13])]
                    co = color_map[np.argmax(scores[option-1][13:23])]
                    att = att_map[np.argmax(scores[option-1][23::])]
                    st.write(f'results: {co} {cat}')
                    st.write(f'Recommendations:')

                    cols = (st.beta_columns([50]*5))
                    for i in range(len(cols)):
                        with cols[i]:
                            st.write(f'[link]({recommend_details.iloc[i, 2]})', unsafe_allow_html=True)
                            image_location = base_dir + recommend_details.iloc[i, 1]
                            st.image(image_location, use_column_width=True)
                
                """Hope you find the recommendations helpful!
                As always, shop responsibly! Think twice before you make a purchase, and don\'t buy
                things you don\'t need.
                Now what? Spread the word and invite your friends and family to join you on this 
                sustainability movement. 
                Every little step counts!"""
                
            else:
                st.markdown('Please try another image.')

if __name__ == "__main__":
    main()
