# %%
import os
import pathlib
import json
import argparse

import cv2
import skvideo.io

# %%
class VideoLabeler:
    def __init__(self, video_path, label_dir):
        frames = skvideo.io.vread(video_path)
        self.frames = frames
        self.i = 0

        # load label state
        pathlib.Path(label_dir).mkdir(parents=True, exist_ok=True)
        self.label_dir = label_dir
        self.label_dict = dict()
        if os.path.isfile(self.json_path):
            self.label_dict = json.load(open(self.json_path, 'r'))

        self.image_buffer = dict()
    
    @property
    def curr_index(self):
        return self.i

    def __len__(self):
        return len(self.frames)
    
    @property
    def json_path(self):
        return os.path.join(self.label_dir, 'labels.json')

    def save_labels(self):
        pathlib.Path(self.label_dir).mkdir(parents=True, exist_ok=True)
        json.dump(self.label_dict, open(self.json_path, 'w'), indent=2)

    def save_images(self):
        pathlib.Path(self.label_dir).mkdir(parents=True, exist_ok=True)
        # glob
        files = pathlib.Path(self.label_dir).glob('*.jpg')
        file_path_map = dict()
        for file in files:
            key = file.stem
            path = str(file.absolute())
            file_path_map[key] = path
        
        # delete unlabeled images
        for key, path in file_path_map.items():
            if key not in self.label_dict:
                os.remove(path)

        # save unsaved images
        for key, img in self.image_buffer.items():
            path = os.path.join(self.label_dir, key + '.jpg')
            cv2.imwrite(path, img[:,:,[2,1,0]])

        self.image_buffer = dict()

    def add_label(self, coord):
        key = str(self.curr_index)
        self.label_dict[key] = coord
        self.image_buffer[key] = self.frames[self.curr_index]

    def delete_label(self):
        key = str(self.curr_index)
        self.label_dict.pop(key, None)
        self.image_buffer.pop(key, None)

    def next_frame(self):
        self.i = min(self.i + 1, len(self.frames) - 1)
        return self.i
    
    def prev_frame(self):
        self.i = max(self.i - 1, 0)
        return self.i
    
    def get_curr_img(self):
        vis_img = self.frames[self.curr_index].copy()
        key = str(self.curr_index)
        if key in self.label_dict:
            coord = self.label_dict[key]
            cv2.drawMarker(vis_img, coord, 
                color=(255,0,0), markerType=cv2.MARKER_CROSS,
                markerSize=20, thickness=1)
        vis_img = vis_img[:,:,[2,1,0]].copy()
        return vis_img

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required=True)
    parser.add_argument('-o', '--output', required=True)
    args = parser.parse_args()

    state = VideoLabeler(args.input, args.output)
    def callback(action, x, y, flags, *userdata):
        if action == cv2.EVENT_LBUTTONDOWN:
            coord = (x,y)
            print(coord)
            state.add_label(coord=coord)
    
    cv2.namedWindow("img", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("img", callback)

    while True:
        cv2.imshow("img", state.get_curr_img())
        key = cv2.waitKey(17)


        if key == ord('q'):
            print('exit')
            break
        elif key == ord('a'):
            print('prev')
            frame = state.prev_frame()
            print(f'{frame}/{len(state)}')
        elif key == ord('d'):
            print('next')
            frame = state.next_frame()
            print(f'{frame}/{len(state)}')
        elif key == 255:
            print('delete')
            state.delete_label()
        elif key == 13:
            print('save')
            state.save_labels()
            state.save_images()
        # if key != -1:
        #     print("key:", key)

# %%
if __name__ == '__main__':
    main()
