# 视频处理
import cv2
import mmcv
import numpy as np
import torch
from mmdet.apis import init_detector, inference_detector
import imageio

src_video_path = "./test_video.mp4"
gif_path = "./color_splash.gif"
des_video_path = "./color_splash.mp4"
model_score_thr = 0.85
model_conf = 'result/mask_rcnn_r50_coco.py'
mode_checkpoint = './result/mask_rcnn_r50_coco_balloon_best.pth'

if __name__ == '__main__':
    video_reader = mmcv.VideoReader(src_video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(des_video_path, fourcc, video_reader.fps, (video_reader.width, video_reader.height))
    model = init_detector(model_conf, mode_checkpoint, device='cuda:0')

    des_frame_list = []

    # 参考 https://github.com/aso538/OpenMMLab_AI_camp_work/blob/main/basic_wor_2/color_splash.py
    count = 1
    for frame in video_reader:
        result = inference_detector(model, frame)
        mask = None
        masks = result[1][0]
        for i in range(len(masks)):
            if result[0][0][i][-1] >= model_score_thr:
                if not mask is None:
                    mask = mask | masks[i]
                else:
                    mask = masks[i]

        # 获取各通道mask像素
        masked_b = frame[:, :, 0] * mask
        masked_g = frame[:, :, 1] * mask
        masked_r = frame[:, :, 2] * mask
        masked = np.concatenate([masked_b[:, :, None], masked_g[:, :, None], masked_r[:, :, None]], axis=2)

        # frame转灰度图
        un_mask = 1 - mask
        frame_b = frame[:, :, 0] * un_mask
        frame_g = frame[:, :, 1] * un_mask
        frame_r = frame[:, :, 2] * un_mask
        frame = np.concatenate([frame_b[:, :, None], frame_g[:, :, None], frame_r[:, :, None]], axis=2).astype(np.uint8)
        frame = mmcv.bgr2gray(frame, keepdim=True)
        frame = np.concatenate([frame, frame, frame], axis=2)

        # mask加到灰度图中
        frame += masked

        # frame = model.show_result(frame, result, score_thr=model_score_thr)

        video_writer.write(frame)
        if count % 100==0:
            # bgr 转 rgb
            frame = frame[..., ::-1]
            des_frame_list.append(frame)
    gif = imageio.mimsave(gif_path, des_frame_list, 'GIF', duration=5 / len(des_frame_list))
    video_writer.release()
    cv2.destroyAllWindows()
