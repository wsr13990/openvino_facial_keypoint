import argparse
import cv2
import numpy as np
from inference import Network

MODEL = 'D:/BELAJAR/OpenVino/facial_keypoint/models/intel/facial-landmarks-35-adas-0002/FP16/facial-landmarks-35-adas-0002.xml'
FACE_CASCADE = 'D:/BELAJAR/OpenVino/facial_keypoint/models/intel/haarcascade_frontalface_default.xml'


def get_args():
    '''
    Gets the arguments from the command line.
    '''
    parser = argparse.ArgumentParser("Run inference on an input video")
    # -- Create the descriptions for the commands
    m_desc = "The location of the model XML file"
    i_desc = "The location of the input file"
    d_desc = "The device name, if not 'CPU'"
    c_desc = "The color of the bounding boxes to draw; RED, GREEN or BLUE"
    ct_desc = "The confidence threshold to use with the bounding boxes"

    # -- Add required and optional groups
    parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')

    # -- Create the arguments
    optional.add_argument("-d", help=d_desc, default='CPU')
    optional.add_argument("-c", help=c_desc, default='BLUE')
    args = parser.parse_args()

    return args


def capture_stream(args):
    # Initialize the Inference Engine
    plugin = Network()
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    # Load the network model into the IE
    plugin.load_model(MODEL, args.d)
    net_input_shape = plugin.get_input_shape()

    # Handle image, video or webcam
    image_flag = False

    # Get and open video capture
    cap = cv2.VideoCapture(0)

    print(net_input_shape)
    width = int(net_input_shape[2])
    height = int(net_input_shape[3])

    out = cv2.VideoWriter('out.mp4', 0x00000021, 30, (width, height))

    # Re-size the frame
    while cap.isOpened():
        cv2.namedWindow("preview")
        flag, frame = cap.read()
        faces = face_cascade.detectMultiScale(frame, 1.2, 2)
        for (x, y, w, h) in faces:
            frame = frame[y:y+h, x:x+w]
        print(frame.shape)
        ori_width = frame.shape[1]
        ori_height = frame.shape[0]
        if not flag:
            break
        key_pressed = cv2.waitKey(60)

        width_ratio = ori_width/width
        height_ratio = ori_height/height
        resized_frame = cv2.resize(frame, (width, height))
        resized_frame = resized_frame.transpose((2, 0, 1))
        resized_frame = resized_frame.reshape(1, *resized_frame.shape)

        plugin.async_inference(resized_frame)

        if plugin.wait() == 0:
            result = plugin.extract_output()
            for i in range(0, result.shape[1], 2):
                x, y = int(
                    result[0][i]*ori_width), int(result[0][i+1]*ori_height)
                cv2.circle(frame, (x, y), 1, (0, 255, 0), 2)
            cv2.imshow("preview", frame)
    # Write out the frame, depending on image or video
        if key_pressed == 27:
            cv2.destroyWindow("preview")
            break
    # Close the stream and any windows at the end of the application
    print(frame)
    out.release()
    cap.release()
    cv2.destroyAllWindows()


def main():
    args = get_args()
    capture_stream(args)


if __name__ == "__main__":
    main()
