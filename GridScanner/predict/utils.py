# written by Xiaohui Zhao
# 2018-01 
# xiaohui.zhao@outlook.com
import numpy as np
import csv
from os.path import join
import os
try:
    import cv2
except ImportError:
    pass

c_threshold = 0.5

def vis_bbox(data_loader, pil_image, grid_table, gt_classes, model_output_val,
         file_name, bboxes, shape, output_filename):
    '''
    Draw rectangle for detected classes on image
    '''

    data_input_flat = grid_table.reshape([-1])
    logits = model_output_val.reshape([-1, data_loader.num_classes])
    bboxes = bboxes.reshape([-1])
    
    max_len = 768*2 # upper boundary of image display size 
    img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    if img is not None:    
        shape = list(img.shape)
        
        bbox_pad = 1
        inf_color = [[255, 250, 240], [152, 245, 255], [119,204,119], [100, 149, 237], 
                    [192, 255, 62], [119,119,204], [114,124,114], [240, 128, 128], [255, 105, 180],
                    [255, 250, 240], [152, 245, 255], [119,204,119], [100, 149, 237], 
                    [192, 255, 62], [119,119,204], [114,124,114], [240, 128, 128], [255, 105, 180],
                    [255, 250, 240], [152, 245, 255], [119,204,119], [100, 149, 237], 
                    [192, 255, 62], [119,119,204], [114,124,114], [240, 128, 128]]
        
        font = cv2.FONT_HERSHEY_COMPLEX
        ft_color = [50, 50, 250]
        
        factor = max_len / max(shape)
        shape[0], shape[1] = [int(s*factor) for s in shape[:2]]
        
        img = cv2.resize(img, (shape[1], shape[0]))        
        overlay_box = np.zeros(shape, dtype=img.dtype)
        overlay_line = np.zeros(shape, dtype=img.dtype)
        for i in range(len(data_input_flat)):
            if len(bboxes[i]) > 0:
                x,y,w,h = [int(p*factor) for p in bboxes[i]]
            else:
                row = i // data_loader.rows
                col = i % data_loader.cols
                x = shape[1] // data_loader.cols * col
                y = shape[0] // data_loader.rows * row
                w = shape[1] // data_loader.cols * 2
                h = shape[0] // data_loader.cols * 2
                    
            if max(logits[i]) > c_threshold:
                inf_id = np.argmax(logits[i])
                if inf_id:
                    try:                
                        cv2.rectangle(overlay_line, (x+bbox_pad,y+bbox_pad), \
                                  (x+bbox_pad+w,y+bbox_pad+h), inf_color[inf_id], max_len//768*2)
                    except:
                        print(inf_id)
        
        # legends
        w = shape[1] // data_loader.cols * 4
        h = shape[0] // data_loader.cols * 2
        for i in range(1, len(data_loader.classes)):
            row = i * 3
            col = 0
            x = shape[1] // data_loader.cols * col
            y = shape[0] // data_loader.rows * row 
            cv2.putText(img, data_loader.classes[i], (x+w,y+h), font, 0.8, ft_color)  
            
            row = i * 3 + 1
            col = 0
            x = shape[1] // data_loader.cols * col
            y = shape[0] // data_loader.rows * row 
            try:
                cv2.rectangle(img, (x+bbox_pad,y+bbox_pad), \
                          (x+bbox_pad+w,y+bbox_pad+h), inf_color[i], max_len//384)   
            except:
                print(i)  
        
        alpha = 0.4
        cv2.addWeighted(overlay_box, alpha, img, 1-alpha, 0, img)
        cv2.addWeighted(overlay_line, 1-alpha, img, 1, 0, img)
        cv2.imwrite(output_filename, img)