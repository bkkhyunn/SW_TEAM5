from flask import Flask, render_template, request
import threading

app = Flask(__name__)
alert_active = False  # 알림 활성화 여부를 나타내는 플래그
alert_lock = threading.Lock()  # 스레드 동기화를 위한 Lock 객체

def show_alert():
    global alert_active

    # 경고음 재생
    # 경고음 파일 경로를 알맞게 수정해주세요
    # 예: alert_sound = "/static/alert.wav"
    alert_sound = "경고음 파일 경로"
    # 경고음 재생 코드

    # 팝업창 표시
    # 팝업창 내용을 알맞게 수정해주세요
    popup_content = "전방에 블랙아이스가 있습니다. 조심하세요."
    # 팝업창 표시 코드

    # 일정 시간(예: 3초) 후에 알림 비활성화
    threading.Timer(3.0, disable_alert).start()

def disable_alert():
    global alert_active
    with alert_lock:
        alert_active = False

@app.route("/process", methods=["POST"])
def process():
    global alert_active
    threshold_pixels = 10  # 임계값 픽셀 개수를 알맞게 수정해주세요

    # 블랙아이스 검출량 확인
    detected_pixels = get_detected_pixels()  # 블랙아이스 검출량을 얻는 함수 호출

    if detected_pixels >= threshold_pixels and not alert_active:
        with alert_lock:
            alert_active = True
        show_alert()

    return "Processed"

if __name__ == "__main__":
    app.run()