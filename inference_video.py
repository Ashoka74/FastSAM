
from ultralytics import YOLO
from utils.tools import *
import argparse
import ast


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path", type=str, default="./weights/FastSAM.pt", help="model"
    )
    parser.add_argument(
        "--img_path", type=str, default="./images/dogs.jpg", help="path to image file"
    )
    parser.add_argument("--imgsz", type=int, default=1024, help="image size")
    parser.add_argument(
        "--iou",
        type=float,
        default=0.9,
        help="iou threshold for filtering the annotations",
    )
    parser.add_argument(
        "--text_prompt", type=str, default=None, help='use text prompt eg: "a dog"'
    )
    parser.add_argument(
        "--conf", type=float, default=0.4, help="object confidence threshold"
    )
    parser.add_argument(
        "--output", type=str, default="./output/", help="image save path"
    )
    parser.add_argument(
        "--randomcolor", type=bool, default=True, help="mask random color"
    )
    parser.add_argument(
        "--point_prompt", type=str, default="[[0,0]]", help="[[x1,y1],[x2,y2]]"
    )
    parser.add_argument(
        "--point_label",
        type=str,
        default="[0]",
        help="[1,0] 0:background, 1:foreground",
    )
    parser.add_argument("--box_prompt", type=str,
                        default="[0,0,0,0]", help="[x,y,w,h]")
    parser.add_argument(
        "--better_quality",
        type=str,
        default=False,
        help="better quality using morphologyEx",
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--device", type=str, default=device, help="cuda:[0,1,2,3,4] or cpu"
    )
    parser.add_argument(
        "--retina",
        type=bool,
        default=True,
        help="draw high-resolution segmentation masks",
    )
    parser.add_argument(
        "--withContours", type=bool, default=False, help="draw the edges of the masks"
    )

    parser.add_argument(
        "--video_path", type=str, default=0, help="path to video file or integer for webcam"
    )

    return parser.parse_args()


def overlay_transparent(frame, mask, alpha=0.5):
    # Convert single channel mask to 3-channel image to match original frame
    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    # Convert the mask to same datatype as original frame
    mask_rgb = mask_rgb.astype(frame.dtype)

    # Blend the original frame and the mask
    overlay = cv2.addWeighted(frame, alpha, mask_rgb, 1-alpha, 0)
    return overlay


def main(args):
    # load model
    model = YOLO(args.model_path)
    args.point_prompt = ast.literal_eval(args.point_prompt)
    args.box_prompt = ast.literal_eval(args.box_prompt)
    args.point_label = ast.literal_eval(args.point_label)
    args.video_path = int(
        args.video_path) if args.video_path.isdigit() else args.video_path

    # Open video stream
    cap = cv2.VideoCapture(args.video_path)
    #cap = cv2.VideoCapture(0)

    frame_count = 0

    while(cap.isOpened()):
        # Read frame
        ret, frame = cap.read()

        # save frame to --img_path
        cv2.imwrite(args.img_path, frame)

        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Apply model to frame
            results = model(
                frame_rgb,
                imgsz=args.imgsz,
                device=args.device,
                retina_masks=args.retina,
                iou=args.iou,
                conf=args.conf,
                max_det=100,
            )

            if args.box_prompt[2] != 0 and args.box_prompt[3] != 0:
                annotations = prompt(results, args, box=True)
                annotations = np.array([annotations])
                fast_process(
                    annotations=annotations,
                    args=args,
                    mask_random_color=args.randomcolor,
                    bbox=convert_box_xywh_to_xyxy(args.box_prompt),
                )

            elif args.text_prompt != None:
                results = format_results(results[0], 0)
                annotations = prompt(results, args, text=True)
                annotations = np.array([annotations])
                fast_process(
                    annotations=annotations, args=args, mask_random_color=args.randomcolor
                )

            elif args.point_prompt[0] != [0, 0]:
                results = format_results(results[0], 0)
                annotations = prompt(results, args, point=True)
                # list to numpy
                annotations = np.array([annotations])
                print(annotations.shape)
                fast_process(
                    annotations=annotations,
                    args=args,
                    mask_random_color=args.randomcolor,
                    points=args.point_prompt,
                )

            else:
                fast_process(
                    annotations=results[0].masks.data,
                    args=args,
                    mask_random_color=args.randomcolor,
                )
        else:
            break

    cv2.imshow("frame", frame)

    # Release capture and destroy windows at the end of the video
    cap.release()
    cv2.destroyAllWindows()


def prompt(results, args, box=None, point=None, text=None):
    ori_img = cv2.imread(args.img_path)
    ori_h = ori_img.shape[0]
    ori_w = ori_img.shape[1]
    if box:
        mask, idx = box_prompt(
            results[0].masks.data,
            convert_box_xywh_to_xyxy(args.box_prompt),
            ori_h,
            ori_w,
        )
    elif point:
        mask, idx = point_prompt(
            results, args.point_prompt, args.point_label, ori_h, ori_w
        )
    elif text:
        mask, idx = text_prompt(results, args)
    else:
        return None
    return mask


if __name__ == "__main__":
    args = parse_args()
    main(args)
