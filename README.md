# TfLiteTools
TfLite tools.


# Test case #
From Movidius MTCNN; links are listed below.
1. https://github.com/movidius/ncappzoo/tree/master/tensorflow/MTCNN
2. https://github.com/ihere1/movidius-face


# metaToTfLite.py # 
metaToTfLite.py is used to tranfer .meta or .pb to tflite. Fill up the config file to suit your case.

ex, --graph_def_file expects for .pb file; if it is provided, the .pb will be used to transfer to .tflite directly.

Otherwise, .meta should be provided; if so, the .meta will be tranfered to .pb and then to .tflite.
Notice that some .pb need to be tranfered internally to avoid **"incompatible with expected float_ref"** issue; please refer to https://github.com/davidsandberg/facenet/issues/161

# runLiteModel.py # 

    parser.add_argument("--tflite", help="tflite's path", default='./test.tflite')
    parser.add_argument("--img", help="img path", default='./test.jpg')
    parser.add_argument("--folder", help="folder path", default=None)
    parser.add_argument("--type", type=int, help="1:float 2:int8", default=1)
    parser.add_argument("--normalize", help="need normalize before input", default=True)
    parser.add_argument("--out", help="output file's path", default='./result.csv')
