import os
import cv2
import torch
import numpy as np
from torchvision import transforms
import torch.nn.functional as F
from PIL import Image, ImageOps
from net.bgnet import Net


def check_video_frame(video_file):
    cap = cv2.VideoCapture(video_file)

    # video info
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    length = frame_count / fps
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    video_info = {'fps':fps, 'frame_count':frame_count, 'length':length, 'width':width, 'height':height}
    print(video_info)

    cap.release()

    return video_info


def split_video_frame(video_file):
    # 영상 프레임 분할 
    imgs = []

    cap = cv2.VideoCapture(video_file)

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        # BGR to RGB 변환
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        imgs.append(frame_rgb)

    cap.release()

    print(f'분할된 프레임 개수 : {len(imgs)}개')
    print('영상 프레임 분할 완료')
    
    return imgs


def run_model(imgs, model_path):
    # sementation model
    segs = []

    # GPU 사용 여부 결정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 모델 가져오기
    model_path = model_path
    model = Net()
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    # input 이미지 전처리
    transform = transforms.Compose([
        transforms.Resize((416,416)),
        transforms.ToTensor(),  # PIL 이미지를 Tensor로 변환
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])  # 이미지 정규화
    ])

    for img in imgs:
        img_pil = Image.fromarray(img)
        # input 이미지 전처리
        input = transform(img_pil).unsqueeze(0).to(device)

        # 모델 실행
        with torch.no_grad():
            _, _, seg, _ = model(input)
        
        # output 이미지 후처리
        output = F.interpolate(seg, size=(720,1280), mode='bilinear', align_corners=False)
        output = output.sigmoid().data.cpu().numpy().squeeze()
        output = (output - output.min()) / (output.max() - output.min() + 1e-8)

        segs.append((output*255).astype(np.uint8))

    print('세그멘테이션 추출 완료')

    return segs


def process_seg(segs, rat=0.15, thr=50):
    # segmentation 제거 : grayscale 0 = 검정색
    # rat : 제거할 pixel 상위 비율 (ratio 0~1 사이의 값)
    # thr : grayscale 임계값 (threshold 0~255 사이의 값)
    processed_segs = []
    seg_ratios = []

    for seg in segs:
        height, width = seg.shape
        top_cutoff = int(height * rat)
        bottom_cutoff = int(height * (1-rat))

        # 1) 상하단 ratio 만큼
        seg[:top_cutoff, :] = 0
        seg[bottom_cutoff:, :] = 0

        # 2) 임계값 미만
        seg[seg < thr] = 0

        processed_segs.append(seg)


        # 세그멘테이션 면적 비율 계산
        road_pixel = (1-2*rat)*height*width
        upthr_pixel = (seg >= thr).sum()  
        seg_ratio = upthr_pixel / road_pixel  

        seg_ratios.append(seg_ratio)

    print('세그멘테이션 처리 완료')

    return processed_segs, seg_ratios


def combine_frame(imgs, processed_segs, alpha=0.2):
    # image + segmentation
    results = []
    img_count = len(imgs)
    seg_count = len(processed_segs)
    print(f'이미지 {img_count}개 - 세그멘테이션 {seg_count}개')

    if img_count == seg_count:
        print(f'프레임과 세그멘테이션 개수가 일치합니다 : {len(imgs)}개')
        for n in range(0,img_count):
            img_pil = Image.fromarray(imgs[n])
            seg_pil = Image.fromarray(processed_segs[n]).convert('L')
            seg_red = ImageOps.colorize(seg_pil, (0,0,0), (255,0,0))
            result = Image.blend(img_pil, seg_red, alpha)
            
            results.append(result)
    else:
        print('프레임과 세그멘테이션 개수가 일치하지 않습니다')

    return results


def reconstruct_video(video_info, results, output_path):
    fps = video_info['fps']
    width = video_info['width']
    height = video_info['height']

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame in results:
        frame_np = np.asarray(frame)
        frame_rgb = cv2.cvtColor(frame_np, cv2.COLOR_BGR2RGB)
        video.write(frame_rgb)

    video.release()