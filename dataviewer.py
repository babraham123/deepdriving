import caffe
from caffe.proto import caffe_pb2
import plyvel
import numpy as np

db = plyvel.DB('../TORCS_baseline_testset/TORCS_Caltech_1F_Testing_280/')

datum = caffe_pb2.Datum.FromString(db.Get('1'))
arr = caffe.io.datum_to_array(datum)
print(datum.label)
print(arr.shape)

print('Keys:')
for key, value in db:
    print(key)
    
db.close()
