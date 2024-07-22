from video_function import check_video_frame, split_video_frame, run_model, process_seg, combine_frame, reconstruct_video

# video info
video_file = "/path/for/video/video.mp4"  # input video 경로 지정
video_info = check_video_frame(video_file)
# encoding error:r''로 escape 처리

# 영상 프레임 분할
imgs = split_video_frame(video_file)

# segmentation model
model_path = "/path/for/pre-trained/model.pth"  # model pth 파일 경로 지정
segs = run_model(imgs, model_path)

# segmentation 가공
processed_segs, seg_ratios = process_seg(segs)

# image + segmentation
results = combine_frame(imgs, processed_segs, 0.2)

# video reconstruction
video_info = video_info
results = results
output_path = "/path/to/save/final_video.mp4"  # video 저장 경로 지정
reconstruct_video(video_info, results, output_path)


# image + segmentation 합치기 전까지 list에 numpy array 형태로 저장되기 떄문에 이미지 확인하기 위한 다음 식 첨부
# from PIL import Image

# # 분할된 원본 이미지 (예시 frame90)
# img_pil = Image.fromarray(imgs[90])
# img_pil.show()