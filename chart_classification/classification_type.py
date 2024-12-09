import os
import numpy as np
import cv2
from tensorflow.keras.models import model_from_json
import settings

class ChartClass:
    def __init__(self) -> None:
        # self.img_path = file_path
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        json_file = open('model/best_model_classification.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.loaded_model = model_from_json(loaded_model_json)
        self.loaded_model.load_weights("model/best_model_classification.h5")
        print("Loaded model from disk")
        
    def classify_charts(self, img_path)->str:
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224))
        image = image / 255.0 

        input_data = np.expand_dims(image, axis=0) 
        pred = self.loaded_model.predict(input_data,
                            verbose=1)
        predicted_class_indices = np.argmax(pred, axis=1)
        
        predicted_chart_type = settings.CHART_TYPES.get(predicted_class_indices[0])
        print(predicted_chart_type)
        
        return predicted_chart_type
    
        
        
        
        
        
    
    