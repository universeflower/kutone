import os
import shutil

file_path = "Sample\Sample\images\\2"
for (path, dir, files) in os.walk(file_path):
    for filename in files:
        ext = os.path.splitext(filename)[-1]
        if ext == '.jpg':
            src = path+'\\' + filename
            print(src)
            dst = 'my_data\\validation\\images'
            shutil.move(src,dst)