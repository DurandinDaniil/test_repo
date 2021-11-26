
import grpc
import numpy as np

from protos.tensorflow.core.framework import tensor_pb2, tensor_shape_pb2, types_pb2
from protos.tensorflow_serving.apis import predict_pb2, prediction_service_pb2_grpc

import cv2
from lattice.app import mask2points
import matplotlib.pyplot as plt
import orjson
import requests
import os

from data_parsing.notebooks.data_parsing import *
from sklearn.metrics import pairwise_distances

from PIL import Image

import owncloud





class Grid(object):
    
    def __init__(self, points, indexes):
        self.points = points
        self.indexes = indexes
        
    def __repr__(self):
        
        return str({'points' : self.points, 'indexes': self.indexes})




class TestTracker():
    
    def __init__(self, roi_corner = None):
        
        self.dots_input_size = tuple((736, 736))
        self.roi_input_size = tuple((224,160))
        
        self.roi_corner = roi_corner
        self.batch_size = 1
        
        self.new_plate = None
        self.prev_frame_hash = None
        self.prev_first_plate = np.array([[2000, 2000], [2000, 2000], [2000, 2000], [2000, 2000]])
        
        
        self.grpc_channel = grpc.insecure_channel(
                "0.0.0.0:8500", options=[('grpc.max_receive_message_length', 2166870)])
        self.stub = prediction_service_pb2_grpc.PredictionServiceStub(self.grpc_channel)
        self.roi_model_address = 'http://localhost:8501/v1/models/roi'
        
    def crop(self, image: np.ndarray, coord: tuple):
        return image[
            coord[1]: coord[1] + self.dots_input_size[0],
            coord[0]: coord[0] + self.dots_input_size[1]
            ]



    def preprocess(self, batch, roi_corner=None):
        tensors = []

        for i in batch:
            try:
                i = self.crop(i, coord=(self.roi_corner))
            except:
                pass
            i = cv2.resize(i, self.dots_input_size).astype(np.float32) / 255.0

            # albumentations.Normalize
            for _, (m, s) in enumerate(zip((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))):
                i[:, :, _] = (i[:, :, _] - m) / s

            i = np.transpose(i, (2, 0, 1))
            tensors.append(i[np.newaxis, :, :, :])

        return np.concatenate(tensors) 
    
    @staticmethod
    def roi_postprocess(mask: np.ndarray):
        mask = (mask > 0.5).astype('uint8')
        conts, h = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
        if len(conts) == 0:
            return None
        cont = sorted(conts, key=lambda z: cv2.contourArea(z))[-1]
        cont = np.squeeze(cont)
        xmax = np.max(cont[:, 0])
        xmin = np.min(cont[:, 0])
        xmin, xmax = xmin / 224 * 1920, xmax / 224 * 1920
        middle = (xmin + xmax) / 2
        return max(int(middle - 736 / 2), 0), 344

    def roi_preprocess(self, batch):
        tensors = []
        for i in batch:
            i = cv2.resize(i, tuple(self.roi_input_size), interpolation=cv2.INTER_AREA)
            k = (i.astype(np.float32) / 255.0).transpose(2, 0, 1)
            tensors.append(np.expand_dims(((k[0] - 0.485) / 0.229), axis=(0, 1)))
        return np.concatenate(tensors)



    def roi_predict(self, batch):

        data = orjson.dumps({"instances": batch},
                            option=orjson.OPT_SERIALIZE_NUMPY)
        headers = {"content-type": "application/json"}
        json_response = requests.post(self.roi_model_address + ':predict',
                                      data=data, headers=headers)
        predictions = np.array(orjson.loads(json_response.text)["predictions"])
        return predictions




    
    def get_prediction(self, path):

        cap = cv2.VideoCapture(path)
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        
        pred_frames = {}
        pred_dots = {}

        for i in range(length):

            ret, frame = cap.read()
            
            
            
            if i % 8 != 0 or frame is None:
                continue
            
            batch = np.expand_dims(frame ,axis=0)
            
            if i == 0:
                roi_batch = self.roi_preprocess(batch)
                roi_pred = self.roi_predict(roi_batch)
                x_crop, y_crop = self.roi_postprocess(roi_pred[-1][0])
                self.roi_corner = tuple((x_crop, y_crop))
                

            batch = self.preprocess(batch, self.roi_corner)
            
            
#             if i > 0:
#                 hash0 = imagehash.average_hash(Image.fromarray(self.crop(frame, coord=(self.roi_corner))))
# #                 print(hash0 - self.prev_frame_hash)
                
#             self.prev_frame_hash = imagehash.average_hash(Image.fromarray(self.crop(frame, coord=(self.roi_corner))))
            
            # save frames
#             pred_frames.update({i : self.crop(frame, self.roi_corner)})

            request = predict_pb2.PredictRequest()
            request.model_spec.name = "dots_detector"
            tensor_shape = tensor_shape_pb2.TensorShapeProto(
                dim=map(lambda _: tensor_shape_pb2.TensorShapeProto.Dim(size=_), [self.batch_size,
                                                        3, self.dots_input_size[0] , self.dots_input_size[1] ]))
            
            request.inputs['input'].CopyFrom(
                tensor_pb2.TensorProto(float_val=batch.flatten(), dtype=types_pb2.DT_FLOAT, tensor_shape=tensor_shape))

            result = self.stub.Predict(request)  # add 3 secs timeout (, 3) as needed
            res = result.outputs["output_0"]
            predicted_mask = np.asarray(res.float_val).reshape(self.dots_input_size)



            grid = mask2points(predicted_mask)
            
            
            
            
            self.prev_frame = grid

            indexes = grid.indexes
            points = grid.points

            pred_dots.update({i: grid})
            
            
            
            
            
#             print(f'suc read {i} frame')
    
        return pred_dots, pred_frames
    
    def get_gt_data(self, path):
        
        data = parse_dict(path)
        
        frames = {}
        for tracks in data['tracks']:
            for track in tracks:
                for fr_id, d in track['frames'].items():
                    fr_id = int(fr_id)
                    if fr_id not in frames.keys():
                        frames[fr_id] = {track['label']: {'points': d['points'], 'type': track['type']}}
                    else:
                        frames[fr_id][track['label']] = {'points': d['points'], 'type': track['type']}

        gt_data = {}

        for frame_id, frame in frames.items():
            try:
                lc = frame['left_chain']['points'][0]
                rc = frame['right_chain']['points'][0]
                z = frame['zigzag']['points'][0]
                verticals = [lc,rc]
                sectors = []


                for i in z.reshape(-1, 4):
                    sectors.append([[i[0],i[1]], [i[2],i[3]]])

                dots = np.array(get_dots_for_verticals_and_sectors(verticals, sectors))



                dots = np.array(list(z) + list(dots))

                gt_data[frame_id] = np.array(dots)
            except Exception as e: 
                pass
        
        return gt_data
    
    def convert_gt_data(self, data):
        converted_gt_data = {}
        for key, value in data.items():
            value = np.array(sorted(value, key = lambda x: x[1], reverse = True))
            
            try:
                value = value.reshape((-1,4,2))
            except:
                continue

            value = np.array([list(sorted(row, key = lambda x: x[0], reverse = False )) for row in value])
            inds = np.array([x for x in range(value.shape[0]*value.shape[1])]).reshape((-1,4))

            value = value.reshape((-1,2))
            x_corr = self.roi_corner[0] 
            y_corr = self.roi_corner[1] 
            value = np.array([(x-x_corr,y-y_corr) for (x,y) in value])

            grid_obj = Grid(value, inds)
            converted_gt_data.update({key : grid_obj})

        return converted_gt_data 
    
    
    # def displacement(self, base: list, to: list) -> list:
    #     return (np.asarray(to, dtype=np.float32) - np.asarray(base, dtype=np.float32)).mean(axis=0).tolist()

    
    
   

    def new_plate_detection(self, first_gt_plate):
        eps = 10
        new_plate = (first_gt_plate[:,1].mean() - self.prev_first_plate[:,1].mean() - eps) > 0
        self.prev_first_plate = first_gt_plate
        return new_plate
    

    def count_metrics_between_events(self, pred_dots, converted_gt_data):

        first_plate_counter = 0
        miss_recognize_first_plate = 0
        frame_per_event = 0
        
        frame_per_event_counter = []
        flag_prev = False
        prev_plate_flag = False
        plate_found = False
        self.prev_first_plate = np.array([[0, 0], [0, 0], [0, 0], [0, 0]])
        for key, value in pred_dots.items():
            pred_plates = value
            try:
                gt_plates = converted_gt_data[key]
            except:
                continue

            if len(gt_plates.indexes) < 2:
                continue
                
                
            if len(pred_plates) > 1:
                distances = pairwise_distances(gt_plates.points[gt_plates.indexes[0]], 
                                               gt_plates.points[gt_plates.indexes[1]], 
                                               metric = 'sqeuclidean') 
                trsh = sum([min(dist) for dist in distances]) / 15
                
            else:
                continue

            new_plate = self.new_plate_detection(gt_plates.points[gt_plates.indexes[0]])

#             print(new_plate)
            distances = pairwise_distances(gt_plates.points[gt_plates.indexes[0]], 
                                       np.array(pred_plates.points)[np.array(pred_plates.indexes)[0]], 
                                        metric = 'sqeuclidean') 

            dist = sum([min(dist) for dist in distances])



            if new_plate:
                if not plate_found:
                    miss_recognize_first_plate += 1
                plate_found = False
                first_plate_counter+=1 
                frame_per_event_counter.append(frame_per_event)
                frame_per_event = 0
                
            else:
                frame_per_event+=1

            if dist < trsh:
                plate_found = True

        result_dict = {
           'First plate Misrecognized' : miss_recognize_first_plate/ first_plate_counter,

           'First plates amount' : first_plate_counter,
           'First plates misrecognized' : miss_recognize_first_plate,
            
            'Mean frame per event' : np.array(frame_per_event_counter).mean()
        }

        return result_dict

        

if __name__ == "__main__":

    videos_path = './vids_data/video'
    annots_path = './vids_data/annotations'

    video_files = os.listdir(videos_path)
    annots_files = os.listdir(annots_path)

    test_tracker = TestTracker()

    metric_by_videos = {}

    url = 'https://ai.ntrlab.ru/mlcloud'
    oc = owncloud.Client(url)
    oc.login('ddurandin', 'ddurandin')

    cloud_videos = oc.list('/EVRAZ.Video-AI.Lava/Healthcheck videoarhive/Workers_DD_DT_Hom_Stitching/vids_data/video')
    cloud_annot = oc.list('/EVRAZ.Video-AI.Lava/Healthcheck videoarhive/Workers_DD_DT_Hom_Stitching/vids_data/annotations')

    for vid,annot in zip(cloud_videos, cloud_annot):
        
        video_file = vid.path.split('/')[-1]
        # download if files doesnt exist
        if vid.path.split('/')[-1] not in video_files:
            oc.get_file(vid.path, './vids_data/video/' + str(vid.path.split('/')[-1]))
                
        if annot.path.split('/')[-1] not in annots_files:
            oc.get_file(annot.path, './vids_data/annotations/' + str(annot.path.split('/')[-1]))
            
        
        
        annot_file = [x for x in annots_files if x[:-4] == video_file[:-4]]
        temp_dict = {}
        temp_dict.update({'vid_path' : videos_path + '/' + video_file, 
                                'annot_path' : annots_path + '/' + annot_file[0]}) 

        pred_dots, pred_frames = test_tracker.get_prediction(videos_path + '/' + video_file)
        gt_data = test_tracker.get_gt_data(annots_path + '/' + annot_file[0])
        converted_gt_data = test_tracker.convert_gt_data(gt_data)
        temp_dict.update({'metrics' :   test_tracker.count_metrics(pred_dots, converted_gt_data)})
        
        metric_by_videos.update({video_file : temp_dict})

    first_plate_misrecognized = 0
    first_plates = 0

    for key, value in metric_by_videos.items():

        
        first_plate_misrecognized += value['metrics']['First plates misrecognized']
        first_plates += value['metrics']['First plates amount']


    with open('tracking_metric.txt', 'w+') as f:
        f.write(str(first_plate_misrecognized / first_plates))