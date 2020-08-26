from predict.clovaa import get_text_boxes
from predict.dataloader.dataloader import DataLoader
from predict.utils import vis_bbox
from config.model_config import model_params
from predict.grpc_client import get_model_output

import os
import numpy as np
import requests
import json
from datetime import datetime


def predict(image, filename):
    '''
    Process Image and get result from cutie model
    '''

    file_name, _ = os.path.splitext(filename)
    file_name = file_name + '_' + str(datetime.now())

    os.mkdir(os.path.join('results', file_name))
    image.save(os.path.join('results', file_name, 'image.png'))

    # Run OCR On Image for Text Boxes
    json_data = get_text_boxes(image, file_name)

    with open(os.path.join('results', file_name, 'ocr.json'), 'w') as fout:
        json.dump(json_data, fout)

    print("OCR done")

    # Parse ocr data for model
    data_loader = DataLoader(json_data, model_params,
                             update_dict=False, load_dictionary=True)

    data = data_loader.fetch_validation_data()

    model_output_val = get_model_output(data)
    model_output_val = np.array(model_output_val)[0]

    shape = data['shape']
    fileName = data['file_name'][0]  # use one single file_name
    bboxes = data['bboxes'][fileName]

    # Draw rectangle for detected classes on image
    vis_bbox(data_loader, image, np.array(data['grid_table'])[0],
             np.array(data['gt_classes'])[0], model_output_val, fileName,
             np.array(bboxes), shape, os.path.join('results', file_name, 'cutie_result.png'))

    logits = model_output_val.reshape([-1, data_loader.num_classes])

    grid_table = np.array(data['grid_table'])[0]
    word_ids = data['word_ids'][fileName]
    data_input_flat = grid_table.reshape([-1])

    c_threshold = 0.5

    final_output = []
    unique_id = []

    # Get class label texts with high confidence
    for i in range(len(data_input_flat)):
        if max(logits[i]) > c_threshold:
            inf_id = np.argmax(logits[i])
            if inf_id and word_ids[i] != []:
                text, bounding_box = idTotext(word_ids[i], json_data)
                if not(word_ids[i] in unique_id):
                    final_output.append({
                        "class_name": data_loader.classes[inf_id],
                        "id": word_ids[i],
                        "text": text,
                        "bounding_box": bounding_box,
                        "confidence": max(logits[i])
                    })
                    unique_id.append(word_ids[i])
                else:
                    for item in range(0, len(final_output)):
                        if(final_output[item]["id"] == word_ids[i]):
                            if final_output[item]["confidence"] < max(logits[i]):
                                final_output[item]["confidence"] = max(
                                    logits[i])

    payload = {
        "model_output": final_output,
        "all_classes": data_loader.classes,
        "filename": file_name
    }

    url = 'http://{host}:{port}/getXLSX'.format(
        host=model_params.invoiceholder_host, port=model_params.invoiceholder_port)

    # Get resullt from Placeholder API
    response = requests.get(url,
                            json=payload
                            )

    response_body = response.content

    # Store result for record
    with open(os.path.join('results', file_name, 'result.xlsx'), 'wb') as fout:
        fout.write(response_body)

    return os.path.join('results', file_name, 'result.xlsx')


def idTotext(id, json_data):
    '''
    Get bbox and box text from box ID.
    '''
    for text_box in json_data['text_boxes']:
        if text_box['id'] == id:
            return text_box['text'], text_box['bbox']

    return None
