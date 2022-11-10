from typing import List, Dict
from sieve.types import Object, StaticObject, FrameFetcher
from sieve.predictors import ObjectPredictor
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import cv2
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

class EmotionPredictor(ObjectPredictor):
    def setup(self):
        model = Sequential()

        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
        model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(1024, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(7, activation='softmax'))

        model.load_weights('model.h5')
        self.model = model

    
    def predict(self, frame_fetcher: FrameFetcher, object: Object) -> StaticObject:
        # Get bounding box from object middle frame
        object_start_frame, object_end_frame = object.start_frame, object.end_frame
        object_temporal = object.get_temporal((object_start_frame + object_end_frame)//2)
        object_bbox = object_temporal.bounding_box
        # Get image from middle frame
        frame_data = frame_fetcher.get_frame((object_start_frame + object_end_frame)//2)
        frame_data = cv2.cvtColor(frame_data, cv2.COLOR_BGR2GRAY)
        # Crop frame data to bounding box
        width = object_bbox.x2 - object_bbox.x1
        height = object_bbox.y2 - object_bbox.y1
        margin = 0.1
        margin_x1 = max(0, int(object_bbox.x1 - (margin * width)))
        margin_y1 = max(0, int(object_bbox.y1 - (margin * height)))
        margin_x2 = min(int(frame_data.shape[1]), int(object_bbox.x2 + (margin * width)))
        margin_y2 = min(int(frame_data.shape[0]), int(object_bbox.y2 + (margin * height)))
        frame_data = frame_data[margin_y1:margin_y2, margin_x1:margin_x2]

        if frame_data.shape[0] == 0 or frame_data.shape[1] == 0:
            ret_data = {"emotion": "unknown", "emotion_confidence": 0}
            return self.get_return_val(object, **ret_data)
        
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(frame_data, (48, 48)), -1), 0)
        prediction = self.model.predict(cropped_img)
        maxindex = int(np.argmax(prediction))
        ret_data = {"emotion": emotion_dict[maxindex], "emotion_confidence": prediction.reshape(-1)[maxindex].item()}
        return self.get_return_val(object, **ret_data)

    # Helper method
    def get_return_val(self, object: Object, **data) -> StaticObject:
        return StaticObject(cls=object.cls, object_id = object.object_id, start_frame=object.start_frame, end_frame=object.end_frame, skip_frames=object.skip_frames, **data)

