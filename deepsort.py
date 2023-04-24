from track_5 import track_data
from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2

object_tracker = DeepSort(max_age=5,
                          n_init=2,
                          nms_max_overlap=1.0,
                          max_cosine_distance=0.3,
                          nn_budget=None,
                          override_track_class=None,
                          embedder="mobilenet",
                          half=True,
                          bgr=True,
                          embedder_gpu=True,
                          embedder_model_name=None,
                          embedder_wts=None,
                          polygon=False,
                          today=None)

data = track_data.copy()

for frame in data:
    img = cv2.imread('/Users/dmitry/Downloads/26.png')
    result = []
    for obj in frame['data']:
        bbc = obj['bounding_box']
        result.append((bbc, 0.8, 'country'))

    tracks = object_tracker.update_tracks(result, frame=img)
    for i, track in enumerate(tracks):
        # if track.is_confirmed():
        #     continue
        frame['data'][i]['track_id'] = track.track_id

print(data)
