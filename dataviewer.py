import caffe
from caffe.proto import caffe_pb2
import plyvel
import numpy as np

db = plyvel.DB('../TORCS_baseline_testset/TORCS_Caltech_1F_Testing_280/')

print('Keys:')
for key, value in db:
    print(key)

datum = caffe_pb2.Datum.FromString(db.get(b'00000001'))
arr = caffe.io.datum_to_array(datum)
print(datum.label)
print(arr.shape)
    
db.close()
