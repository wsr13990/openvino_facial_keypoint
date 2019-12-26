import argparse
import cv2
import numpy as np
from inference import Network

face_detection_xml = 'D:/BELAJAR/OpenVino/facial_keypoint/models/intel/face-detection-adas-0001/FP16/face-detection-adas-0001.xml'
facial_keypoint_xml = 'D:/BELAJAR/OpenVino/facial_keypoint/models/intel/facial-landmarks-35-adas-0002/FP16/facial-landmarks-35-adas-0002.xml'
FACE_CASCADE = 'D:/BELAJAR/OpenVino/facial_keypoint/models/intel/haarcascade_frontalface_default.xml'
CPU_EXTENSION = 'D:/Program Files (x86)/IntelSWTools/openvino_2019.3.379/inference_engine/samples/intel64/Release/cpu_extension.dll'

def get_args():
    '''
    Gets the arguments from the command line.
    '''
    parser = argparse.ArgumentParser("Run inference on an input video")
    # -- Create the descriptions for the commands
    d_desc = "The device name, if not 'CPU'"
    c_desc = "The color of the bounding boxes to draw; RED, GREEN or BLUE"
    lw_desc = "Haarcascader Line width"
    kw_desc = "Keypoint line width"

    # -- Add required and optional groups
    parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')

    # -- Create the arguments
    optional.add_argument("-d", help=d_desc, default='CPU')
    optional.add_argument("-c", help=c_desc, default='BLUE')
    optional.add_argument("-lw", help=lw_desc, default=1)
    optional.add_argument("-kw", help=kw_desc, default=2)
    args = parser.parse_args()

    return args

def resize(x1,x2,y1,y2,height,width):
    x1 = int(x1*width)
    x2 = int(x2*width)
    y1 = int(y1*height)
    y2 = int(y2*height)
    return (x1,x2,y1,y2)

def capture_stream(args):
    if args.c == 'YELLOW':
        color = (0, 255, 0)
    elif args.c == 'BLUE':
        color = (255, 0, 0)
    elif args.c == 'RED':
        color = (0, 0, 255)

    # Initialize the Inference Engine
    f_det = Network()
    f_keypts = Network()
    # face_cascade = cv2.CascadeClassifier(
    #     cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    # Load the network model into the IE
    f_det.load_model(face_detection_xml, args.d, cpu_extension=CPU_EXTENSION)
    f_keypts.load_model(facial_keypoint_xml, args.d, cpu_extension=CPU_EXTENSION)
    f_det_input_shape = f_det.get_input_shape()
    f_keypts_input_shape = f_keypts.get_input_shape()

    # Handle image, video or webcam
    image_flag = False

    # Get and open video capture
    cap = cv2.VideoCapture(0)

    # print(net_input_shape)
    f_det_width = int(f_det_input_shape[2])
    f_det_height = int(f_det_input_shape[3])

    f_keypts_width = int(f_keypts_input_shape[2])
    f_keypts_height = int(f_keypts_input_shape[3])

    #Video writer
    # out = cv2.VideoWriter('out.mp4', 0x00000021, 30, (width,height))

    # Re-size the frame
    while cap.isOpened():
        cv2.namedWindow("preview")
        flag, frame = cap.read()
        # faces = face_cascade.detectMultiScale(frame, 1.2, 2)
        
        image = cv2.resize(frame, (f_det_width, f_det_height))
        image = image.transpose((2, 1, 0))
        image = image.reshape(1, *image.shape)
        image_height, image_width = image.shape[-2:]
        f_det.async_inference(image)
        if f_det.wait() == 0:
            faces = f_det.extract_output()
        if faces is not None:
            for (a, b, conf, x1, y1, x2, y2) in [faces.squeeze()[0]]:
                x1_resize,x2_resize,y1_resize,y2_resize = resize(
                    x1,x2,y1,y2,frame.shape[0],frame.shape[1])
                face_image = frame[y1_resize:y2_resize,x1_resize:x2_resize]
                face_height, face_width = [int(i) for i in face_image.shape[:2]]
                # Draw facial rectangle
                x1_resize,x2_resize,y1_resize,y2_resize = resize(
                    x1,x2,y1,y2,frame.shape[0],frame.shape[1])
                frame = cv2.rectangle(
                    frame, (x1_resize, y1_resize), (x2_resize, y2_resize), color, int(args.lw))
                face_image = cv2.resize(face_image, (f_keypts_width, f_keypts_height))
                face_image = face_image.transpose((2, 0, 1))
                face_image = face_image.reshape(1, *face_image.shape)

                # Perform async inference
                f_keypts.async_inference(face_image)
                if f_keypts.wait() == 0:
                    result = f_keypts.extract_output()
                    for i in range(0, result.shape[1], 2):
                        x, y = int(
                            x1_resize+result[0][i]*face_width), int(y1_resize+result[0][i+1]*face_height)
                        # Draw Facial key points
                        cv2.circle(frame, (x, y), 1, color, int(args.kw))
        if not flag:
            break
        key_pressed = cv2.waitKey(60)
        # out.write(frame)
        cv2.imshow("preview", frame)
    # Write out the frame, depending on image or video
        if key_pressed == 27:
            cv2.destroyWindow("preview")
            break
    # Close the stream and any windows at the end of the application
    print(frame)
    cap.release()
    # out.release()
    cv2.destroyAllWindows()


def main():
    args = get_args()
    capture_stream(args)


if __name__ == "__main__":
    main()
