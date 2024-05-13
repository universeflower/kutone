import json
import os

def get_class_id(annotation):
    class_mapping = {
        '금속캔': 0,
        '종이': 1,
        '페트병': 2,
        '플라스틱': 3,
        '스티로폼': 4,
        '비닐': 5,
        '유리병': 6,
        '페트병_이물질': 7,
        '플라스틱_이물질': 8,
        '페트병_찌그러진': 9,
        '플라스틱_찌그러진': 10
    }
    
    class_name = annotation['CLASS']
    dirtiness = annotation['DIRTINESS']
    damage = annotation['DAMAGE']

    if class_name in ['페트병', '플라스틱'] and dirtiness == "이물질 (외부)":
        class_name += '_이물질'
    elif class_name in ['페트병', '플라스틱'] and damage != "원형":
        class_name += '_찌그러진'
    
    return class_mapping[class_name]

# JSON 파일 로드
file_path = "Sample\Sample\labels\\2"
for (path, dir, files) in os.walk(file_path):
    for filename in files:
        ext = os.path.splitext(filename)[-1]
        if ext == '.json':
            with open(path+'\\' + filename, 'r', encoding='UTF8') as file:
                print(path+'\\' + filename)
                data = json.load(file)
                image_info = data['IMAGE_INFO']
                image_width = image_info['IMAGE_WIDTH']
                image_height = image_info['IMAGE_HEIGHT']
                image_file_name = image_info['FILE_NAME'].split('.')[0]

                annotations = data['ANNOTATION_INFO']

                for annotation in annotations:
                    if annotation['SHAPE_TYPE'] == 'BOX':
                        points = annotation['POINTS'][0]
                        x_min = points[0]
                        y_min = points[1]
                        box_width = points[2]
                        box_height = points[3]
                        x_max = x_min + box_width
                        y_max = y_min + box_width
    
                        x_center = (x_min + x_max / 2) / image_width
                        y_center = (y_min + y_max / 2) / image_height
                        width = box_width / image_width
                        height = box_height / image_height
                    elif annotation['SHAPE_TYPE'] == 'POLYGON':
                        points = annotation['POINTS']
                        img_width = 1920
                        img_height = 1080

           
                        min_x = min(point[0] for point in points)
                        max_x = max(point[0] for point in points)
                        min_y = min(point[1] for point in points)
                        max_y = max(point[1] for point in points)


                        bbox_width = max_x - min_x
                        bbox_height = max_y - min_y

                        x_center = (min_x + max_x)/2
                        y_center = (min_y +max_y) / 2


                        x_center = x_center / img_width
                        y_center = y_center / img_height
                        width = bbox_width / img_width
                        height = bbox_height / img_height
                    else:
                        continue

    
                    class_id = get_class_id(annotation)
                    yolo_format = f"{class_id} {x_center} {y_center} {width} {height}\n"
    
                    with open('my_data\\validation\labels\\' + f"{image_file_name}.txt", 'w') as yolo_file:
                        yolo_file.write(yolo_format)

                    print("변환 완료!")

