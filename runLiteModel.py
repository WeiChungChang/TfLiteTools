import numpy as np
import tensorflow as tf
import cv2
import os
import sys
import argparse

def runModel(interpreter, input_data, dataType):
    # Get input and output tensors.
    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    #print('input_details = ', input_details)
    #print('output_details = ', output_details)
    #print('type(input_data) = ', type(input_data), ' ', input_data.shape)
    input_data = np.expand_dims(input_data, axis=0)
    # change the following line to feed into your own data.

    interpreter.set_tensor(input_details[0]['index'], input_data)

    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    output_data = output_data.astype(float)
    if dataType == 2:
        quant = output_details[0]['quantization']
        output_data = output_data.astype(float)
        output_data = (output_data - quant[1]) * quant[0]
    return output_data

def readImg(img_path):
    image = cv2.imread(img_path)
    im_data = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return im_data

def liteRunImgFile(model_path, img_path, isFloat = 1, needNormalize = False):
    # Load TFLite mo(del and allocate tensors.
    interpreter = tf.contrib.lite.Interpreter(model_path)
    interpreter.allocate_tensors()

    im_data = readImg(img_path);
    if isFloat == 1:
        im_data = np.float32(im_data)
        if needNormalize == True:
            im_data = (im_data - 127.5) * 0.0078125
    else:
        #im_data = np.int(im_data)
        im_data = np.uint8(im_data)

    output_data, = runModel(interpreter, im_data, isFloat)

    return output_data

def liteRunFolder(model_path, input_dir, isFloat = 1, output = './result.csv'):
    #output_data = []
    score_file = open(output, "w")
    files = os.listdir(input_dir)
    for f in sorted(files):
        f = input_dir + '/' + f
        #print(f, ': is float = ', isFloat)
        score = liteRunImgFile(model_path, f, isFloat)
        score = score.flatten()
        sort_ = score.sort()
        for i in range(0, len(score)):
            score_file.write('%f %f\n' % (score[i], sort_[i]))
        
#liteRunFolder('./result.tflite', './out', False)

def main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--tflite", help="tflite's path", default='./test.tflite')
    parser.add_argument("--img", help="img path", default='./test.jpg')
    parser.add_argument("--folder", help="folder path", default=None)
    parser.add_argument("--type", type=int, help="1:float 2:int8", default=1)
    parser.add_argument("--normalize", help="need normalize before input", default=True)
    parser.add_argument("--out", help="output file's path", default='./result.csv')
    args = parser.parse_args()
    print("====================================================")
    print("******************* run tflit **********************")
    print("====================================================")
    print("tflite's path: \t" + str(args.tflite))
    print("img path: \t" + str(args.img))
    print("folder path: \t" + str(args.folder))
    print("type \t" + str(args.type))
    print("normalize: \t" + str(args.normalize))
    print("out: \t" + str(args.out))
    print(args.tflite)

    outfile = open(args.out, "w")
    if args.type == 1:
        prefix = 'f'
    else:
        prefix = 'i'	
    outfile.write('index, %s_val\n' % (prefix))
    if args.folder == None:        
        out = liteRunImgFile(args.tflite, args.img, args.type, args.normalize)
        out = out.flatten()
        sort_ = np.sort(out)
        #print(type(sort_), type(out), out.sort())
        for i in range(0, len(out)):
            outfile.write('%d, %f\n' % (i, out[i]))
    else:
        liteRunFolder(args.tflite, args.folder, args.type, args.out)
    #print(args.tflite)
    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))