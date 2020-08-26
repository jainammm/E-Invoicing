def getMaxConfidence(model_output, class_name):
    outputs = []

    for item in model_output:
        if item['class_name'] == class_name:
            outputs.append(item)

    confidence = 0
    final_output = ''
    final_id = -1

    for item in outputs:
        if confidence < item['confidence']:
            final_output = item['text']
            final_id = item['id']
            confidence = item['confidence']

    return final_id, final_output


def getAddress(model_output, class_name):
    outputs = []

    for item in model_output:
        if item['class_name'] == class_name:
            outputs.append(item)

    confidence = 0
    bounding_box = []

    for item in outputs:
        if confidence < item['confidence']:
            bounding_box = item['bounding_box']
            confidence = item['confidence']

    if bounding_box == []:
        return ''

    word_height = bounding_box[3] - bounding_box[1]

    valid_box_ymin = bounding_box[1] - int(2.5*word_height)
    valid_box_ymax = bounding_box[3] + int(2.5*word_height)

    final_address = ''

    for item in outputs:
        bbox_item = item['bounding_box']
        ymin = bbox_item[1]
        ymax = bbox_item[3]
        if ymin < valid_box_ymax and ymax > valid_box_ymin:
            final_address += item['text'] + ' '

    return final_address


def getTotalAmount(model_output, class_name):
    outputs = []

    for item in model_output:
        if item['class_name'] == class_name:
            outputs.append(item)

    confidence = 0
    bounding_box = []

    for item in outputs:
        if confidence < item['confidence']:
            bounding_box = item['bounding_box']
            confidence = item['confidence']

    if bounding_box == []:
        return ''

    word_height = bounding_box[3] - bounding_box[1]

    valid_box_ymin = bounding_box[1] - int(0.25*word_height)
    valid_box_ymax = bounding_box[3] + int(0.25*word_height)

    final_amount = ''

    for item in outputs:
        bbox_item = item['bounding_box']
        ymin = bbox_item[1]
        ymax = bbox_item[3]
        if ymax < valid_box_ymax and ymin > valid_box_ymin:
            final_amount += item['text'] + ' '

    return final_amount


def getTable(model_output, class_names):
    base_class = 'HSN'

    bboxes_row = []
    bounding_box_hsn = []

    for item in model_output:
        if item['class_name'] == base_class:
            bounding_box_hsn.append(item['bounding_box'])

    for i, item in enumerate(bounding_box_hsn):
        word_height = item[3] - item[1]

        if i == 0:
            bbox_row_ymin = item[1] - (1 * word_height)
        else:
            bbox_row_ymin = item[1] - int(0.25 * word_height)

        if i == len(bounding_box_hsn) - 1:
            bbox_row_ymax = item[3] + (2*word_height)
        else:
            bbox_row_ymax = max(
                bounding_box_hsn[i+1][1] - (0.2 * word_height), item[3] + int(0.25 * word_height))

        bboxes_row.append([bbox_row_ymin, bbox_row_ymax])

    final_rows = []

    for bbox in bboxes_row:
        row_dict = {}
        for class_name in class_names:
            if class_name == 'TITLE':
                output = ''
                for item in model_output:
                    if item['class_name'] == class_name:
                        ymin, ymax = item['bounding_box'][1], item['bounding_box'][3]
                        if ymin < bbox[1] and ymax > bbox[0]:
                            output += item['text'] + ' '

                row_dict[class_name] = output
            else:
                output = []
                for item in model_output:
                    if item['class_name'] == class_name:
                        ymin, ymax = item['bounding_box'][1], item['bounding_box'][3]
                        if ymin < bbox[1] and ymax > bbox[0]:
                            output.append(item)

                _, item_value = getMaxConfidence(output, class_name)
                row_dict[class_name] = item_value

        final_rows.append(row_dict)

    return final_rows
