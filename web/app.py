import os
from flask import Flask, request, render_template, jsonify, Response, redirect
# import sys
# sys.path.append(r'C:/yolo/yolov12-main/')
from ultralytics import YOLO
from PIL import Image
import numpy as np
from datetime import datetime
import requests
import random
import json as pyjson
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.fonts import addMapping
import logging

app = Flask(__name__)

# 关闭浏览器缓存
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

# 获取所有模型文件夹及其下的.pt模型
MODELS_BASE_DIR = r"D:\BTD\yoloservice\models"
MODEL_FOLDERS = [
    d for d in os.listdir(MODELS_BASE_DIR)
    if os.path.isdir(os.path.join(MODELS_BASE_DIR, d))
]
MODEL_DICT = {}
for folder in MODEL_FOLDERS:
    folder_path = os.path.join(MODELS_BASE_DIR, folder)
    pt_files = [f for f in os.listdir(folder_path) if f.endswith('.pt')]
    if pt_files:
        MODEL_DICT[folder] = pt_files

# 默认选择
DEFAULT_FOLDER = MODEL_FOLDERS[0] if MODEL_FOLDERS else None
DEFAULT_MODEL = MODEL_DICT[DEFAULT_FOLDER][0] if DEFAULT_FOLDER else None
model = YOLO(os.path.join(MODELS_BASE_DIR, DEFAULT_FOLDER, DEFAULT_MODEL) if DEFAULT_MODEL else None)

# 设置上传与检测目录
UPLOAD_FOLDER = './web/static/uploads'
DETECT_FOLDER = './web/static/detections'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DETECT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['DETECT_FOLDER'] = DETECT_FOLDER

# SiliconFlow（硅基流动）大模型API配置
SILICONFLOW_API_URL = "https://api.siliconflow.cn/v1/chat/completions"  # 硅基流动
SILICONFLOW_API_KEY = "你的密钥"  # 请替换为你的实际密钥

# 日志配置
LOG_PATH = './web/logs/web.log'
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[logging.FileHandler(LOG_PATH, encoding='utf-8'), logging.StreamHandler()]
)

logger = logging.getLogger(__name__)

# 替换诊断函数，适配新API

def get_deepseek_diagnosis(detections, lang="zh"):
    """
    调用 SiliconFlow Qwen/QwQ-32B 大模型API，生成更智能的诊断结论。
    detections: 检测结果列表，如 ["cat: 98.5%", "dog: 95.2%"]
    lang: 诊断语言
    """
    import re
    prompt = (
        f"你是一名权威医学影像诊断专家，请根据以下目标检测结果，直接给出具体、权威的医学诊断结论，并提出针对性的医学建议：\n"
        f"检测结果：{detections}\n"
        f"要求：1. 先给出明确的医学诊断结论（如疾病名称、异常描述等）；2. 给出针对性的医学建议（如是否需要进一步检查、随访、治疗等）；3. 语言简明、专业、权威。"
    )
    payload = {
        "model": "Qwen/QwQ-32B",
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }
    headers = {
        "Authorization": f"Bearer {SILICONFLOW_API_KEY}",
        "Content-Type": "application/json"
    }
    logger.info(f"[AI分析-请求] detections={detections}")
    try:
        resp = requests.post(SILICONFLOW_API_URL, json=payload, headers=headers, timeout=60)
        if resp.status_code == 200:
            result = resp.json()
            # 兼容Qwen返回格式
            if "choices" in result and result["choices"] and "message" in result["choices"][0]:
                content = result["choices"][0]["message"]["content"].strip()
                # 去除markdown符号和多余格式
                content = re.sub(r'^[#>*\-\d\.\s]+', '', content, flags=re.MULTILINE)  # 去除行首符号
                content = re.sub(r'[`*_>\[\]#\-]', '', content)  # 去除常见markdown符号
                content = content.replace('**', '').replace('__', '')
                logger.info(f"[AI分析-成功] result={content}")
                return content.strip()
            else:
                logger.error(f"[AI分析-异常格式] result={result}")
                return str(result)
        else:
            logger.error(f"[AI分析-失败] status={resp.status_code}, text={resp.text}")
            return f"[SiliconFlow诊断失败] {resp.status_code}: {resp.text}"
    except Exception as e:
        logger.error(f"[AI分析-异常] {e}")
        return f"[SiliconFlow诊断异常] {e}。请稍后重试，或检查网络/API平台状态。"

RECORD_FILE = './web/static/records.json'

def save_record(record):
    try:
        records = []
        if os.path.exists(RECORD_FILE):
            with open(RECORD_FILE, 'r', encoding='utf-8') as f:
                try:
                    records = pyjson.load(f)
                except Exception:
                    records = []
        records.insert(0, record)
        with open(RECORD_FILE, 'w', encoding='utf-8') as f:
            pyjson.dump(records, f, ensure_ascii=False, indent=2)
        logger.info(f"[记录保存] record_id={record.get('record_id')}, patient={record.get('patient_name')}")
    except Exception as e:
        logger.error(f"[记录保存-异常] {e}")

def get_record_by_id(record_id):
    if os.path.exists(RECORD_FILE):
        with open(RECORD_FILE, 'r', encoding='utf-8') as f:
            try:
                records = pyjson.load(f)
                for r in records:
                    if r.get('record_id') == record_id:
                        return r
            except Exception:
                pass
    return None

def get_stat_data():
    """统计首页所需数据，并返回最近6条图片信息"""
    stat_total = 0
    stat_today = 0
    stat_ai = 0
    stat_trend = []
    recent_images = []
    if os.path.exists(RECORD_FILE):
        with open(RECORD_FILE, 'r', encoding='utf-8') as f:
            try:
                records = pyjson.load(f)
            except Exception:
                records = []
        stat_total = len(records)
        from datetime import date
        today_str = date.today().strftime('%Y-%m-%d')
        stat_today = sum(1 for r in records if r.get('time', '').startswith(today_str))
        stat_ai = sum(1 for r in records if r.get('ai_suggestion') and r.get('ai_suggestion') != '未检测到目标，无法分析。')
        # 统计最近7天趋势
        from collections import Counter
        trend_counter = Counter()
        for r in records:
            t = r.get('time', '')[:10]
            if t:
                trend_counter[t] += 1
        import datetime
        days = [ (datetime.date.today() - datetime.timedelta(days=i)).strftime('%Y-%m-%d') for i in range(6, -1, -1)]
        stat_trend = [ {'date': d, 'count': trend_counter.get(d, 0)} for d in days ]
        # 最近6条图片信息
        for r in records[:6]:
            if r.get('image_path'):
                recent_images.append({
                    'image_path': r['image_path'],
                    'time': r.get('time', ''),
                    'patient_name': r.get('patient_name', ''),
                    'ai_diagnosis': r.get('ai_diagnosis', '')
                })
    return stat_total, stat_today, stat_ai, stat_trend, recent_images

@app.route('/history')
def history_page():
    """历史检测记录页面"""
    if os.path.exists(RECORD_FILE):
        with open(RECORD_FILE, 'r', encoding='utf-8') as f:
            try:
                records = pyjson.load(f)
            except Exception:
                records = []
    else:
        records = []
    return render_template('history.html', records=records)

@app.route('/history/delete/<record_id>', methods=['POST'])
def delete_history_record(record_id):
    """删除指定检测记录"""
    import threading
    lock = threading.Lock()
    with lock:
        if os.path.exists(RECORD_FILE):
            with open(RECORD_FILE, 'r', encoding='utf-8') as f:
                try:
                    records = pyjson.load(f)
                except Exception:
                    records = []
            new_records = [r for r in records if r.get('record_id') != record_id]
            with open(RECORD_FILE, 'w', encoding='utf-8') as f:
                pyjson.dump(new_records, f, ensure_ascii=False, indent=2)
            return jsonify({'success': True})
    return jsonify({'success': False, 'msg': '未找到记录'}), 404

@app.route('/', methods=['GET', 'POST'])
def upload_detect():
    selected_folder = request.form.get('model_folder') if request.method == 'POST' else DEFAULT_FOLDER
    selected_model = request.form.get('model') if request.method == 'POST' else (MODEL_DICT[selected_folder][0] if selected_folder else None)
    global model
    if selected_folder and selected_model:
        model_path = os.path.join(MODELS_BASE_DIR, selected_folder, selected_model)
        if not hasattr(model, 'model_path') or getattr(model, 'model_path', None) != model_path:
            model = YOLO(model_path)

    if request.method == 'POST':
        # 获取客户端上传的图片和病人信息
        patient_name = request.form.get('patient_name', '')
        patient_gender = request.form.get('patient_gender', '')
        patient_age = request.form.get('patient_age', '')
        patient_id = request.form.get('patient_id', '')
        image_file = request.files["image"]
        if image_file:
            filename = datetime.now().strftime("%Y%m%d%H%M%S") + "_" + image_file.filename
            upload_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            detect_path = os.path.join(app.config["DETECT_FOLDER"], filename)
            image_file.save(upload_path)
            # 使用YOLOv12进行目标检测
            results = model(upload_path)
            # 绘制检测结果图像并保存
            result_img_array = results[0].plot()
            result_pil = Image.fromarray(result_img_array)
            result_pil.save(detect_path)
            # 提取检测框信息（标签 + 置信度）
            detections = []
            ai_diagnosis = ""
            boxes = results[0].boxes
            if boxes is not None and boxes.cls.numel() > 0:
                class_count = {}
                for cls_id, conf in zip(boxes.cls, boxes.conf):
                    class_name = model.names[int(cls_id)]
                    confidence = round(float(conf) * 100, 2)
                    detections.append(f"{class_name}: {confidence}%")
                    class_count[class_name] = class_count.get(class_name, 0) + 1
                diagnosis_list = [f"检测到{count}个{cls_name}" for cls_name, count in class_count.items()]
                ai_diagnosis = "，".join(diagnosis_list) + "。"
            else:
                detections.append("No objects detected.")
                ai_diagnosis = "未检测到目标，请更换图片或调整拍摄角度。"
            # 检测时不再自动生成AI分析
            ai_suggestion = ""  # 或 "未分析"
            # 生成唯一检测单号
            record_id = datetime.now().strftime('%Y%m%d%H%M%S') + str(random.randint(1000,9999))
            # 保存记录
            record = {
                'record_id': record_id,
                'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'patient_name': patient_name,
                'patient_gender': patient_gender,
                'patient_age': patient_age,
                'patient_id': patient_id,
                'image_path': f'detections/{filename}',
                'detections': detections,
                'ai_diagnosis': ai_diagnosis,
                'ai_suggestion': ai_suggestion
            }
            save_record(record)
            stat_total, stat_today, stat_ai, stat_trend, recent_images = get_stat_data()
            return render_template(
                'index.html',
                prediction="Detection Complete",
                detections=detections,
                ai_diagnosis=ai_diagnosis,
                deepseek_diagnosis=ai_suggestion,
                image_path=f"detections/{filename}",
                model_folders=MODEL_FOLDERS,
                model_dict=MODEL_DICT,
                selected_folder=selected_folder,
                selected_model=selected_model,
                record_id=record_id,
                patient_name=patient_name,
                patient_gender=patient_gender,
                patient_age=patient_age,
                patient_id=patient_id,
                stat_total=stat_total,
                stat_today=stat_today,
                stat_ai=stat_ai,
                stat_trend=stat_trend,
                recent_images=recent_images
            )
    # GET请求或未上传时
    stat_total, stat_today, stat_ai, stat_trend, recent_images = get_stat_data()
    return render_template(
        'index.html',
        prediction=None,
        model_folders=MODEL_FOLDERS,
        model_dict=MODEL_DICT,
        selected_folder=selected_folder,
        selected_model=selected_model,
        stat_total=stat_total,
        stat_today=stat_today,
        stat_ai=stat_ai,
        stat_trend=stat_trend,
        recent_images=recent_images
    )

@app.route('/detect', methods=['GET', 'POST'])
def detect_page():
    selected_folder = request.form.get('model_folder') if request.method == 'POST' else DEFAULT_FOLDER
    selected_model = request.form.get('model') if request.method == 'POST' else (MODEL_DICT[selected_folder][0] if selected_folder else None)
    global model
    if selected_folder and selected_model:
        model_path = os.path.join(MODELS_BASE_DIR, selected_folder, selected_model)
        if not hasattr(model, 'model_path') or getattr(model, 'model_path', None) != model_path:
            model = YOLO(model_path)
    if request.method == 'POST':
        patient_name = request.form.get('patient_name', '')
        patient_gender = request.form.get('patient_gender', '')
        patient_age = request.form.get('patient_age', '')
        patient_id = request.form.get('patient_id', '')
        image_file = request.files["image"]
        if image_file:
            filename = datetime.now().strftime("%Y%m%d%H%M%S") + "_" + image_file.filename
            upload_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            detect_path = os.path.join(app.config["DETECT_FOLDER"], filename)
            image_file.save(upload_path)
            results = model(upload_path)
            result_img_array = results[0].plot()
            result_pil = Image.fromarray(result_img_array)
            result_pil.save(detect_path)
            detections = []
            ai_diagnosis = ""
            boxes = results[0].boxes
            if boxes is not None and boxes.cls.numel() > 0:
                class_count = {}
                for cls_id, conf in zip(boxes.cls, boxes.conf):
                    class_name = model.names[int(cls_id)]
                    confidence = round(float(conf) * 100, 2)
                    detections.append(f"{class_name}: {confidence}%")
                    class_count[class_name] = class_count.get(class_name, 0) + 1
                diagnosis_list = [f"检测到{count}个{cls_name}" for cls_name, count in class_count.items()]
                ai_diagnosis = "，".join(diagnosis_list) + "。"
            else:
                detections.append("No objects detected.")
                ai_diagnosis = "未检测到目标，请更换图片或调整拍摄角度。"
            # 检测时不再自动生成AI分析
            ai_suggestion = ""  # 或 "未分析"
            record_id = datetime.now().strftime('%Y%m%d%H%M%S') + str(random.randint(1000,9999))
            record = {
                'record_id': record_id,
                'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'patient_name': patient_name,
                'patient_gender': patient_gender,
                'patient_age': patient_age,
                'patient_id': patient_id,
                'image_path': f'detections/{filename}',
                'detections': detections,
                'ai_diagnosis': ai_diagnosis,
                'ai_suggestion': ai_suggestion
            }
            save_record(record)
            return render_template(
                'detect.html',
                prediction="Detection Complete",
                detections=detections,
                ai_diagnosis=ai_diagnosis,
                deepseek_diagnosis=ai_suggestion,
                image_path=f"detections/{filename}",
                model_folders=MODEL_FOLDERS,
                model_dict=MODEL_DICT,
                selected_folder=selected_folder,
                selected_model=selected_model,
                record_id=record_id,
                patient_name=patient_name,
                patient_gender=patient_gender,
                patient_age=patient_age,
                patient_id=patient_id
            )
    return render_template('detect.html', prediction=None, model_folders=MODEL_FOLDERS, model_dict=MODEL_DICT, selected_folder=selected_folder, selected_model=selected_model)

@app.route('/ai_analyze', methods=['POST'])
def ai_analyze():
    data = request.get_json()
    detections = data.get('detections', [])
    record_id = data.get('record_id')
    if not detections:
        return jsonify({'result': '未获取到检测结果'}), 400
    result = get_deepseek_diagnosis(detections)
    # 写回历史记录
    if record_id:
        try:
            if os.path.exists(RECORD_FILE):
                with open(RECORD_FILE, 'r', encoding='utf-8') as f:
                    records = pyjson.load(f)
                for r in records:
                    if r.get('record_id') == record_id:
                        r['ai_suggestion'] = result
                        break
                with open(RECORD_FILE, 'w', encoding='utf-8') as f:
                    pyjson.dump(records, f, ensure_ascii=False, indent=2)
                logging.info(f"[AI分析结果写回] record_id={record_id}")
        except Exception as e:
            logging.error(f"[AI分析结果写回异常] record_id={record_id}, error={e}")
    return jsonify({'result': result})

FONT_PATH = os.path.join(os.path.dirname(__file__), '../yoloservice/utils/LXGWWenKai-Bold.ttf')
if os.path.exists(FONT_PATH):
    pdfmetrics.registerFont(TTFont('WenKai', FONT_PATH))
    addMapping('WenKai', 0, 0, 'WenKai')

@app.route('/report/pdf/<record_id>')
def report_pdf(record_id):
    record = get_record_by_id(record_id)
    if not record:
        logging.error(f"[PDF导出-未找到记录] record_id={record_id}")
        return '未找到该检测记录', 404
    from io import BytesIO
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.enums import TA_LEFT
    styles = getSampleStyleSheet()
    zh_title = ParagraphStyle('zhTitle', parent=styles['Title'], fontName='WenKai', fontSize=22, alignment=TA_LEFT)
    zh_h3 = ParagraphStyle('zhH3', parent=styles['Heading3'], fontName='WenKai', fontSize=14)
    zh_normal = ParagraphStyle('zhNormal', parent=styles['Normal'], fontName='WenKai', fontSize=12)
    elements = []
    title = Paragraph("医学影像AI检测报告", zh_title)
    elements.append(title)
    elements.append(Spacer(1, 18))
    info_data = [
        ["检测单号", record['record_id']],
        ["检测时间", record['time']],
        ["患者姓名", record['patient_name']],
        ["性别", record['patient_gender']],
        ["年龄", record['patient_age']],
        ["检查号", record['patient_id']],
    ]
    table = Table(info_data, hAlign='LEFT', colWidths=[80, 300])
    table.setStyle(TableStyle([
        ('FONTNAME', (0,0), (-1,-1), 'WenKai'),
        ('BACKGROUND', (0,0), (-1,0), colors.lightblue),
        ('TEXTCOLOR', (0,0), (-1,0), colors.black),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('FONTSIZE', (0,0), (-1,-1), 11),
        ('BOTTOMPADDING', (0,0), (-1,0), 8),
        ('BACKGROUND', (0,1), (-1,-1), colors.whitesmoke),
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
    ]))
    elements.append(table)
    elements.append(Spacer(1, 18))
    elements.append(Paragraph("检测结果：", zh_h3))
    for d in record['detections']:
        elements.append(Paragraph(d, zh_normal))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph("AI诊断结论：", zh_h3))
    elements.append(Paragraph(record['ai_diagnosis'], zh_normal))
    elements.append(Spacer(1, 12))
    if record.get('ai_suggestion'):
        elements.append(Paragraph("AI智能建议：", zh_h3))
        elements.append(Paragraph(record['ai_suggestion'], zh_normal))
    # 添加免责声明
    elements.append(Spacer(1, 24))
    # 创建醒目的免责声明样式
    disclaimer_style = ParagraphStyle(
        'disclaimer',
        parent=zh_normal,
        fontName='WenKai',
        fontSize=14,
        textColor=colors.red,
        backColor=colors.lightyellow,
        borderColor=colors.red,
        borderWidth=2,
        borderPadding=12,
        alignment=1,  # 居中对齐
        spaceAfter=8,
        spaceBefore=8
    )
    disclaimer = " 免责声明 <br/><br/>本报告中的AI分析建议仅供临床参考，不能替代专业医生的诊断和治疗决策。如有疑问请咨询专业医务人员。"
    elements.append(Paragraph(disclaimer, disclaimer_style))
    try:
        doc.build(elements)
        pdf = buffer.getvalue()
        buffer.close()
        from flask import make_response
        response = make_response(pdf)
        response.headers['Content-Type'] = 'application/pdf'
        response.headers['Content-Disposition'] = f'attachment; filename=AI_Report_{record_id}.pdf'
        logging.info(f"[PDF导出-成功] record_id={record_id}")
        return response
    except Exception as e:
        logging.error(f"[PDF导出-异常] record_id={record_id}, error={e}")
        return 'PDF导出失败', 500

@app.route('/history/search', methods=['POST'])
def search_history_records():
    """搜索历史检测记录"""
    data = request.get_json()
    search_name = data.get('name', '').strip()
    search_gender = data.get('gender', '').strip()
    search_age = data.get('age', '').strip()
    search_date = data.get('date', '').strip()
    
    if os.path.exists(RECORD_FILE):
        with open(RECORD_FILE, 'r', encoding='utf-8') as f:
            try:
                records = pyjson.load(f)
            except Exception:
                records = []
    else:
        records = []
    
    # 过滤记录
    filtered_records = []
    for record in records:
        match = True
        
        # 姓名匹配（模糊搜索）
        if search_name and search_name.lower() not in record.get('patient_name', '').lower():
            match = False
        
        # 性别匹配
        if search_gender and search_gender != record.get('patient_gender', ''):
            match = False
        
        # 年龄匹配
        if search_age and search_age != record.get('patient_age', ''):
            match = False
        
        # 日期匹配（匹配检测日期）
        if search_date and search_date not in record.get('time', ''):
            match = False
        
        if match:
            filtered_records.append(record)
    
    logging.info(f"[历史记录搜索] 搜索条件: name={search_name}, gender={search_gender}, age={search_age}, date={search_date}, 结果数量: {len(filtered_records)}")
    return jsonify({'records': filtered_records, 'total': len(filtered_records)})

if __name__ == '__main__':
    app.run(debug=True)
