import requests
from io import BytesIO


def ocr(image_data):
    '''
    Get OCR result from clovaa-ai
    '''

    multipart_form_data = {
        'image': ('0.jpg', image_data),
    }

    url = 'https://clova.ai/ocr/api/general/ko/recognition?ts=1595701921417&s=3894416e780dee0611a09e1a7e4c3830b60b3897'

    response = requests.post(url,
                             files=multipart_form_data)

    return response.json()['annotations']


def get_text_boxes(image, filename):
    '''
    Convert ocr output for model input
    '''

    b = BytesIO()
    image.save(b, 'PNG')
    data = ocr(b.getvalue())

    width, height = image.size

    final_data = {}
    final_data['text_boxes'] = []
    final_data['fields'] = []
    final_data["global_attributes"] = {
        "file_id": filename
    }

    for count, ob in enumerate(data):
        bbox = []

        xmin, xmax, ymin, ymax = 0, 0, 0, 0

        if 'boundingPoly' in ob:
            for box_num, box in enumerate(ob['boundingPoly']):
                # print(box_num, box)
                if(box_num == 0):
                    xmin = int(width * float(box[0]))
                    ymin = int(height * float(box[1]))
                    bbox.append(xmin)
                    bbox.append(ymin)

                if(box_num == 2):
                    xmax = int(width * float(box[0]))
                    ymax = int(height * float(box[1]))
                    bbox.append(xmax)
                    bbox.append(ymax)

        if 'text' in ob:
            final_data['text_boxes'].append({
                'id': count,
                'bbox': bbox,
                'text': ob['text']
            })

    return final_data
