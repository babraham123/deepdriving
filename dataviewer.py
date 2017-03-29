import caffe
from caffe.proto import caffe_pb2
import plyvel
import numpy as np
import cv2

db = plyvel.DB('../TORCS_baseline_testset/TORCS_Caltech_1F_Testing_280/')
# db = plyvel.DB('../TORCS_Training_1F/')

print('Keys:')
for key, ~ in db:
    print(key)

datum = caffe_pb2.Datum.FromString(db.get(b'00000001'))
arr = caffe.io.datum_to_array(datum)
print([i for i in datum.float_data])
print(arr.shape)

cv2.imwrite('color_img.jpg', arr)
    
db.close()

# datum = caffe.proto.caffe_pb2.Datum()
# datum.ParseFromString(value)

# datum.add_float_data(shared->angle);
# datum.add_float_data(shared->toMarking_L);
# datum.add_float_data(shared->toMarking_M);
# datum.add_float_data(shared->toMarking_R);
# datum.add_float_data(shared->dist_L);
# datum.add_float_data(shared->dist_R);
# datum.add_float_data(shared->toMarking_LL);
# datum.add_float_data(shared->toMarking_ML);
# datum.add_float_data(shared->toMarking_MR);
# datum.add_float_data(shared->toMarking_RR);
# datum.add_float_data(shared->dist_LL);
# datum.add_float_data(shared->dist_MM);
# datum.add_float_data(shared->dist_RR);
# datum.add_float_data(shared->fast);
