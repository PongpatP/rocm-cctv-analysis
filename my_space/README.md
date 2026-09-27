# ROCm CCTV AI Analysis - Study & Development Space (`my_space`)

เอกสารและชุดโค้ดสำหรับศึกษาและพัฒนา Pipeline การวิเคราะห์กล้องวงจรปิด (CCTV Analytics) โดยอ้างอิงจากสถาปัตยกรรมหลักของระบบที่ทำงานบน AMD ROCm (MI300X)

---

## 1. ตารางวิเคราะห์ Task, Tool และ License

| Task | Tool | License | บทบาทและการทำงานในระบบ |
|---|---|:---:|---|
| **Detection** | RT-DETR (Baidu / PaddleDetection) | Apache-2.0 | ตรวจจับวัตถุ 80 COCO classes แบบ Real-time (เน้น 5 คลาสหลัก: person, car, truck, motorcycle, bicycle) โดยเป็นสถาปัตยกรรม Transformer ที่เป็น **NMS-free** ช่วยให้รันแบบ Static graph บน AMD MIGraphX ได้อย่างมีเสถียรภาพ |
| **Tracker** | Clean-room BoT-SORT | MIT | ติดตามวัตถุภายในกล้องเดียวกัน (In-Camera Tracking) โดยใช้ Kalman Filter ในระนาบ `[cx, cy, w, h]` และการจับคู่ 2 ขั้นตอน (High-conf แล้วตามด้วย Low-conf) เขียนขึ้นใหม่แบบ Clean-room เพื่อเลี่ยงข้อจำกัดของสัญญาอนุญาต AGPL-3.0 |
| **Per-frame ReID** | osnet_x0_25 (torchreid) | MIT | สกัดเวกเตอร์เอกลักษณ์ขนาด 512 มิติ (L2-normalized) ต่อเนื่องในแต่ละเฟรม เพื่อนำมารวมกับ IoU ในการจับคู่วัตถุ ช่วยป้องกันปัญหา Track ID สลับเวลาคนเดินสวนกันหรือถูกบดบังชั่วคราว |
| **Cross-camera ReID** | YoutuReID | Apache-2.0 | สกัดเวกเตอร์เอกลักษณ์ขนาด 768 มิติ สำหรับเชื่อมโยงบุคคลข้ามกล้อง (Global ID) โดยมี Appearance Bank เก็บ Centroids และ Exemplars พร้อมคัดกรองเส้นทางด้วย Camera Reachability Graph |
| **Face recognition** | CVLface (MSU CVLab) | MIT | ตรวจจับและจัดแนวใบหน้าด้วย Landmark (MediaPipe/YuNet) แล้วสกัดเวกเตอร์ขนาด 512 มิติด้วยโมเดล CVLface เพื่อเปรียบเทียบกับ Gallery บุคคลที่ลงทะเบียนไว้ล่วงหน้า |
| **Plate detection** | RF-DETR (Rikkosse/rfdetr_licences_plate_detector) | Apache-2.0 | ตรวจจับตำแหน่งป้ายทะเบียนรถยนต์/รถจักรยานยนต์จากภาพ Crop ของยานพาหนะ โดยประมวลผลเฉพาะกล้องบริเวณทางเข้า-ออก |
| **Plate OCR** | PaddleOCR / PaddleX | Apache-2.0 | อ่านตัวอักษรและตัวเลขบนป้ายทะเบียน พร้อมระบบ Multi-frame Voting (`PlateTracker`) เพื่อสรุปผลทะเบียนที่แม่นยำที่สุดเมื่อรถแล่นผ่าน |
| **LLM / VLM** | Gemma 4 | Google Gemma Terms of Use (Free commercial) | VLM ขนาด 31B ทำหน้าที่สกัด Semantic Attributes (สีเสื้อผ้า, ประเภทกระเป๋า, สี/ทรงของรถ) ให้มนุษย์ตรวจสอบได้ และเป็นสมองกลของระบบ AI Investigator ที่สืบค้นข้อมูลในฐานข้อมูลด้วย Fat Tools |
| **Inference runtime** | ONNX Runtime + MIGraphX | MIT | Runtime หลักที่แปลง ONNX Graph ให้รันบน AMD ROCm ผ่าน Execution Provider `MIGraphX` พร้อมระบบ Cache ไฟล์ `.mxr` เพื่อลดเวลา Compile |
| **Infrastructure** | ROCm, Docker, go2rtc | Various open source | แพลตฟอร์มฮาร์ดแวร์ GPU AMD, ระบบ Containerize แบบ Microservices, และ Video Gateway ดึง RTSP จากกล้องมากระจายต่อ |

---

## 2. โครงสร้างไฟล์ใน `my_space`

- [`requirements.txt`](file:///c:/01_MyFolder/03_Code/01_Git/rocm-cctv-analysis/my_space/requirements.txt): รายการไลบรารี Python สำหรับการทดสอบและพัฒนา
- [`01_detection.ipynb`](file:///c:/01_MyFolder/03_Code/01_Git/rocm-cctv-analysis/my_space/01_detection.ipynb): ศึกษาการใช้งาน RT-DETR และการกรองคลาส
- [`02_tracker_and_reid.ipynb`](file:///c:/01_MyFolder/03_Code/01_Git/rocm-cctv-analysis/my_space/02_tracker_and_reid.ipynb): ศึกษา BoT-SORT Kalman Filter ร่วมกับ OSNet x0.25
- [`03_cross_camera_reid.ipynb`](file:///c:/01_MyFolder/03_Code/01_Git/rocm-cctv-analysis/my_space/03_cross_camera_reid.ipynb): ศึกษาการผูก Global ID ข้ามกล้องด้วย YoutuReID และ Reachability Graph
- [`04_face_recognition.ipynb`](file:///c:/01_MyFolder/03_Code/01_Git/rocm-cctv-analysis/my_space/04_face_recognition.ipynb): ศึกษา Face Landmark Alignment และ CVLface Cosine Matching
- [`05_plate_pipeline.ipynb`](file:///c:/01_MyFolder/03_Code/01_Git/rocm-cctv-analysis/my_space/05_plate_pipeline.ipynb): ศึกษาการตรวจจับป้ายด้วย RF-DETR และอ่านข้อความด้วย PaddleOCR
- [`06_vlm_reasoning.ipynb`](file:///c:/01_MyFolder/03_Code/01_Git/rocm-cctv-analysis/my_space/06_vlm_reasoning.ipynb): ศึกษาการเรียกใช้ Gemma 4 สำหรับ Attribute Extraction และ Grounding Gate
- [`integrated_pipeline.py`](file:///c:/01_MyFolder/03_Code/01_Git/rocm-cctv-analysis/my_space/integrated_pipeline.py): สคริปต์ Python ตัวอย่างการเชื่อมต่อทุกโมดูลเข้าด้วยกันเป็น Pipeline เดียว

---

## 3. ขั้นตอนการเตรียมสภาพแวดล้อมและการรัน (Conda Environment)

### ขั้นตอนที่ 1: สร้างและเปิดใช้งาน Conda Environment
```bash
conda create -n cctv-analysis python=3.10 -y
conda activate cctv-analysis
```

### ขั้นตอนที่ 2: ติดตั้ง Dependencies
```bash
cd c:\01_MyFolder\03_Code\01_Git\rocm-cctv-analysis\my_space
pip install -r requirements.txt
```

> **หมายเหตุเรื่อง Hardware Runtime:**
> - หากรันบนเครื่องเซิร์ฟเวอร์ที่มี GPU AMD และ ROCm: สามารถติดตั้ง `onnxruntime-migraphx` เพื่อเร่งความเร็วบนชิป MI300X
> - หากรันบนเครื่องทั่วไปหรือไม่มีไฟล์โมเดล `.onnx`: โค้ดทั้งหมดได้รับการออกแบบให้มี **Simulation Fallback Mode** สามารถรันผ่าน CPU ได้ทันทีเพื่อการศึกษาทดลอง

### ขั้นตอนที่ 3: การรัน Jupyter Notebook
เปิด Jupyter Lab เพื่อทดลองทีละโมดูล:
```bash
jupyter lab
```
จากนั้นเปิดไฟล์ [`01_detection.ipynb`](file:///c:/01_MyFolder/03_Code/01_Git/rocm-cctv-analysis/my_space/01_detection.ipynb) ถึง [`06_vlm_reasoning.ipynb`](file:///c:/01_MyFolder/03_Code/01_Git/rocm-cctv-analysis/my_space/06_vlm_reasoning.ipynb)

### ขั้นตอนที่ 4: การรัน Integrated Pipeline
ทดสอบการทำงานเชื่อมโยงทุกโมดูล:
```bash
python integrated_pipeline.py
```
สคริปต์จะประมวลผลเฟรมจำลองและแสดงผลลัพธ์เป็นโครงสร้าง JSON ที่ผูกข้อมูล Bounding Box, Track ID, Global ID, Face Recognition, License Plate, และ VLM Semantic Attributes เข้าด้วยกันอย่างสมบูรณ์
