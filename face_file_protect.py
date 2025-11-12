import tkinter as tk
from tkinter import filedialog, messagebox, ttk, simpledialog
import os
import threading
import pickle
import numpy as np
import base64
import time
from Crypto.Cipher import AES
from Crypto.Protocol.KDF import PBKDF2
from Crypto.Random import get_random_bytes
import cv2
import insightface
from insightface.app import FaceAnalysis
import csv
from datetime import datetime

# ---------------------- 初始化人脸识别工具 ----------------------
face_app = FaceAnalysis(name='buffalo_l')
face_app.prepare(ctx_id=0, det_size=(640, 640))

# ---------------------- 人脸特征与字符串互转工具函数 ----------------------
def feat_to_str(feat):
    feat_bytes = feat.tobytes()
    return base64.b64encode(feat_bytes).decode('utf-8')

def str_to_feat(feat_str):
    feat_bytes = base64.b64decode(feat_str.encode('utf-8'))
    return np.frombuffer(feat_bytes, dtype=np.float32)

# ---------------------- 文件夹处理工具函数 ----------------------
def get_all_files_in_folder(folder_path):
    """递归获取文件夹内所有文件的绝对路径和相对路径"""
    all_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            # 跳过隐藏文件（避免加密系统文件）
            if file.startswith('.'):
                continue
            abs_path = os.path.join(root, file)
            rel_path = os.path.relpath(abs_path, folder_path)
            all_files.append((abs_path, rel_path))
    return all_files

def create_folder_if_not_exists(folder_path):
    """创建文件夹（如果不存在）"""
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

# ---------------------- 日志与关联信息工具函数 ----------------------
# 日志存储路径（自动创建logs文件夹）
LOG_FOLDER = "encryptor_logs"
create_folder_if_not_exists(LOG_FOLDER)
LOG_FILE_PATH = os.path.join(LOG_FOLDER, "operation_logs.csv")

def init_log_file():
    """初始化日志文件（如果不存在）"""
    if not os.path.exists(LOG_FILE_PATH):
        with open(LOG_FILE_PATH, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # 日志字段：时间、操作类型、操作内容、关联对象、状态、备注
            writer.writerow(["操作时间", "操作类型", "操作内容", "关联对象", "操作状态", "备注"])

def write_log(operation_type, operation_content, related_obj, status, remark=""):
    """写入操作日志"""
    init_log_file()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]  # 保留毫秒
    with open(LOG_FILE_PATH, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([now, operation_type, operation_content, related_obj, status, remark])

def export_logs(start_time=None, end_time=None, export_path=None):
    """导出日志（支持时间筛选）"""
    if not os.path.exists(LOG_FILE_PATH):
        return False, "无日志数据可导出"
    
    # 读取所有日志
    logs = []
    with open(LOG_FILE_PATH, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        logs.append(header)
        for row in reader:
            # 时间筛选（格式：2025-01-01 12:00:00.000）
            if start_time or end_time:
                log_time = row[0]
                log_dt = datetime.strptime(log_time, "%Y-%m-%d %H:%M:%S.%f")
                if start_time and log_dt < datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S"):
                    continue
                if end_time and log_dt > datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S"):
                    continue
            logs.append(row)
    
    # 选择导出路径
    if not export_path:
        export_path = filedialog.asksaveasfilename(
            title="导出日志",
            defaultextension=".csv",
            filetypes=[("CSV文件", "*.csv")]
        )
        if not export_path:
            return False, "取消导出"
    
    # 写入导出文件
    with open(export_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(logs)
    return True, f"日志已导出至：{export_path}"

def parse_encrypted_file_info(file_path, known_features, known_names):
    """解析加密文件的关联信息"""
    try:
        with open(file_path, 'rb') as f:
            # 读取原有字段
            salt = f.read(16)
            nonce = f.read(16)
            tag = f.read(16)
            stored_feat_bytes = f.read(512 * 4)
            stored_feat = np.frombuffer(stored_feat_bytes, dtype=np.float32)
            
            # 尝试读取新增的关联信息（兼容旧版文件）
            try:
                # 新增字段：人脸名称长度（4字节）→ 人脸名称 → 加密时间（8字节）→ 加密算法（16字节）→ 密钥有效期（4字节）
                name_len = int.from_bytes(f.read(4), byteorder='little')
                face_name = f.read(name_len).decode('utf-8')
                encrypt_timestamp = int.from_bytes(f.read(8), byteorder='little')
                encrypt_algorithm = f.read(16).decode('utf-8').strip('\x00')  # 去除填充符
                key_valid_days = int.from_bytes(f.read(4), byteorder='little')
            except:
                # 旧版文件无新增字段，返回基础信息
                return {
                    "status": "success",
                    "is_old_version": True,
                    "face_name": "未知（旧版文件）",
                    "encrypt_time": "未知（旧版文件）",
                    "encrypt_algorithm": "AES-GCM（默认）",
                    "key_valid_days": "无（旧版文件）",
                    "similarity": "需验证后查看"
                }
            
            # 计算人脸相似度（与已录入人脸匹配）
            similarity = "无匹配人脸"
            if len(known_features) > 0:
                sim_scores = [np.dot(stored_feat, feat) for feat in known_features]
                max_sim = max(sim_scores) if sim_scores else 0
                if max_sim > 0.85:
                    match_name = known_names[sim_scores.index(max_sim)]
                    similarity = f"{match_name}（{max_sim:.4f}）"
                else:
                    similarity = f"无匹配（最高{max_sim:.4f}）"
            
            # 格式化时间
            encrypt_time = datetime.fromtimestamp(encrypt_timestamp).strftime("%Y-%m-%d %H:%M:%S")
            
            return {
                "status": "success",
                "is_old_version": False,
                "face_name": face_name,
                "encrypt_time": encrypt_time,
                "encrypt_algorithm": encrypt_algorithm,
                "key_valid_days": f"{key_valid_days}天" if key_valid_days > 0 else "永久",
                "similarity": similarity
            }
    except Exception as e:
        return {
            "status": "error",
            "error_msg": f"解析失败：{str(e)}"
        }

# ---------------------- 加载人脸数据函数 ----------------------
def load_face_data():
    known_features = []
    known_names = []
    enroll_timestamps = []
    if os.path.exists("insight_face_db.pkl"):
        with open("insight_face_db.pkl", "rb") as f:
            data = pickle.load(f)
            if len(data) == 2:
                known_features, known_names = data
                file_ctime = os.path.getctime("insight_face_db.pkl")
                enroll_timestamps = [file_ctime for _ in known_names]
            else:
                known_features, known_names, enroll_timestamps = data
    return np.array(known_features), known_names, enroll_timestamps

# ---------------------- 人脸录入功能 ----------------------
def enroll_face_gui(name, callback):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("错误", "无法打开摄像头！")
        callback(None)
        return

    print(f"=== 录入 {name} 的人脸 ===")
    face_feat = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 镜像处理：水平翻转
        frame = cv2.flip(frame, 1)

        faces = face_app.get(frame)
        if len(faces) == 1:
            face = faces[0]
            bbox = face.bbox.astype(int)
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            cv2.putText(frame, f"Enrolling: {name}", (bbox[0], bbox[1]-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, "Press 's' to save, 'q' to cancel", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        cv2.imshow("Face Enrollment", frame)
        key = cv2.waitKey(1)

        if key == ord('s') and len(faces) == 1:
            face_feat = faces[0].embedding
            face_feat = face_feat / np.linalg.norm(face_feat)
            print(f"√ {name} 人脸录入成功！")
            break
        elif key == ord('q'):
            print("× 取消录入")
            break

    cap.release()
    cv2.destroyAllWindows()
    callback(face_feat, int(time.time()))

# ---------------------- 人脸验证功能 ----------------------
def verify_face(target_feat, callback):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("错误", "无法打开摄像头！")
        callback(False)
        return

    print("=== 开始人脸验证 ===")
    verified = False
    threshold = 0.7

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 镜像处理：水平翻转
        frame = cv2.flip(frame, 1)

        faces = face_app.get(frame)
        for face in faces:
            curr_feat = face.embedding
            curr_feat = curr_feat / np.linalg.norm(curr_feat)
            
            sim_score = np.dot(target_feat, curr_feat)
            bbox = face.bbox.astype(int)
            
            if sim_score > threshold:
                color = (0, 255, 0)
                text = f"Match Success! Similarity: {sim_score:.2f}"
                verified = True
            else:
                color = (0, 0, 255)
                text = f"Match Failed! Similarity: {sim_score:.2f}"

            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            cv2.putText(frame, text, (bbox[0], bbox[1]-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.putText(frame, "Press 'q' to exit verification", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        cv2.imshow("Face Verification (Real-time)", frame)

        if cv2.waitKey(1) == ord('q'):
            break
        if verified:
            break

    cap.release()
    cv2.destroyAllWindows()
    callback(verified)

# ---------------------- 文件加密工具主类（优化版） ----------------------
class FaceFileEncryptorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("人脸识别文件加密工具（葛文凯22122862）")
        self.root.geometry("850x650")
        self.root.minsize(750, 550)
        
        # 优化后的配色方案
        self.COLORS = {
            'primary': "#16409C",      # 主色：适中深蓝（保持专业感，不刺眼）
            'secondary': "#64748B",    # 辅助色：浅蓝（交互反馈更柔和）
            'accent': "#0C3198EA",       # 强调色：深一点的蓝（关键操作突出但不突兀）
            'danger': '#EF4444',       # 危险色：稍柔的红（警示明确但不刺眼）
            'success': '#10B981',      # 成功色：温和绿（反馈清晰）
            'warning': '#F59E0B',      # 警告色：暖橙（保留提醒性）
            'background': '#F0F2F5',   # 主背景：极浅灰（比纯白暗5%，减少反光，类似微信/办公软件背景）
            'card_bg': '#FFFFFF',      # 卡片背景：纯白（与主背景形成轻微对比，突出内容区域）
            'text_primary': '#1E293B', # 主文本：深灰黑（与浅灰背景对比度足够，清晰可读）
            'text_secondary': '#64748B',# 次文本：中灰（与主文本区分，不抢眼）
            'border': '#E2E8F0'        # 边框色：浅灰（弱化边缘，避免割裂感）
        }
        
        # 加载人脸数据
        self.known_features, self.known_names, self.enroll_timestamps = load_face_data()
        self.selected_face_feat = None  # 加密时选中的人脸特征
        
        # 文件夹加密相关变量
        self.selected_folder = None  # 选中的待加密文件夹
        self.encrypt_folder_save_path = None  # 加密后文件夹保存路径
        self.folder_files = []  # 文件夹内所有文件（绝对路径, 相对路径）
        
        self.setup_ui_fonts()
        self.setup_custom_styles()  # 自定义控件样式
        self.create_main_layout()
        
        self.encrypt_files = []
        self.decrypt_files = []
        self.tmp_key_path = None  # 临时密钥文件路径
        self.is_processing = False

    def setup_ui_fonts(self):
        """设置全局字体"""
        self.root.option_add("*Font", ('Microsoft YaHei', 11))
        self.root.configure(bg=self.COLORS['background'])

    def setup_custom_styles(self):
        """自定义控件样式"""
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # 基础样式
        self.style.configure('.', 
                           background=self.COLORS['background'],
                           foreground=self.COLORS['text_primary'],
                           bordercolor=self.COLORS['border'],
                           darkcolor=self.COLORS['secondary'],
                           lightcolor=self.COLORS['background'])
        
        # 标签框架样式
        self.style.configure("TLabelFrame",
                           background=self.COLORS['background'],
                           foreground=self.COLORS['text_primary'],
                           bordercolor=self.COLORS['border'],
                           relief=tk.FLAT,
                           padding=10)
        self.style.configure("TLabelFrame.Label",
                           font=('Microsoft YaHei', 11, 'bold'),
                           background=self.COLORS['background'],
                           foreground=self.COLORS['primary'],
                           padding=(0, 0, 10, 5))
        
        # 按钮样式
        self.style.configure("Primary.TButton",
                           background=self.COLORS['primary'],
                           foreground='white',
                           bordercolor=self.COLORS['primary'],
                           relief=tk.FLAT,
                           padding=8)
        self.style.map("Primary.TButton",
                      background=[('active', self.COLORS['secondary']), ('pressed', self.COLORS['primary'])],
                      foreground=[('active', 'white'), ('pressed', 'white')])
        
        self.style.configure("Danger.TButton",
                           background=self.COLORS['danger'],
                           foreground='white',
                           bordercolor=self.COLORS['danger'],
                           relief=tk.FLAT,
                           padding=8)
        self.style.map("Danger.TButton",
                      background=[('active', '#B91C1C'), ('pressed', self.COLORS['danger'])],
                      foreground=[('active', 'white'), ('pressed', 'white')])
        
        self.style.configure("Secondary.TButton",
                           background=self.COLORS['card_bg'],
                           foreground=self.COLORS['text_primary'],
                           bordercolor=self.COLORS['border'],
                           relief=tk.FLAT,
                           padding=8)
        self.style.map("Secondary.TButton",
                      background=[('active', '#F1F5F9'), ('pressed', self.COLORS['card_bg'])])
        
        # 进度条样式
        self.style.configure("Custom.Horizontal.TProgressbar",
                           background=self.COLORS['accent'],
                           troughcolor=self.COLORS['border'],
                           bordercolor=self.COLORS['border'],
                           relief=tk.FLAT,
                           thickness=8)
        self.style.map("Custom.Horizontal.TProgressbar",
                      background=[('active', self.COLORS['primary'])])
        
        # 下拉框样式
        self.style.configure("TCombobox",
                           background=self.COLORS['card_bg'],
                           foreground=self.COLORS['text_primary'],
                           bordercolor=self.COLORS['border'],
                           relief=tk.FLAT,
                           padding=6)
        self.style.map("TCombobox",
                      background=[('active', '#F1F5F9')],
                      fieldbackground=[('readonly', self.COLORS['card_bg'])])
        
        # Treeview样式
        self.style.configure("Custom.Treeview",
                           background=self.COLORS['card_bg'],
                           foreground=self.COLORS['text_primary'],
                           bordercolor=self.COLORS['border'],
                           relief=tk.FLAT,
                           rowheight=24)
        self.style.configure("Custom.Treeview.Heading",
                           background=self.COLORS['primary'],
                           foreground='white',
                           font=('Microsoft YaHei', 11, 'bold'),
                           padding=8)
        self.style.map("Custom.Treeview",
                      background=[('selected', self.COLORS['secondary']), ('active', '#F1F5F9')],
                      foreground=[('selected', 'white')])
        
        # 标签页样式
        self.style.configure("TNotebook",
                           background=self.COLORS['border'],
                           borderwidth=0,
                           padding=5)
        self.style.configure("TNotebook.Tab",
                           background=self.COLORS['border'],
                           foreground=self.COLORS['text_secondary'],
                           font=('Microsoft YaHei', 12, 'bold'),
                           borderwidth=0,
                           width=10,
                           anchor='center')
        
        self.style.map("TNotebook.Tab",
                      background=[('selected', self.COLORS['primary']), ('active', '#E2E8F0')],
                      foreground=[('selected', 'white'), ('active', self.COLORS['text_primary'])])

    def create_main_layout(self):
        """创建主布局"""
        # 主容器
        main_container = tk.Frame(self.root, 
                                 bg=self.COLORS['background'],
                                 relief=tk.FLAT,
                                 borderwidth=0)
        main_container.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        # 标题栏
        title_frame = tk.Frame(main_container, bg=self.COLORS['background'])
        title_frame.pack(fill=tk.X, padx=20, pady=(20, 15))
        tk.Label(title_frame, 
                text="人脸识别文件加密工具", 
                font=('Microsoft YaHei', 16, 'bold'),
                bg=self.COLORS['background'],
                fg=self.COLORS['primary']).pack(side=tk.LEFT)
        
        # 标签页
        self.notebook = ttk.Notebook(main_container, style="TNotebook")
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 20))
        
        self.encrypt_frame = ttk.Frame(self.notebook, style="TFrame")
        self.decrypt_frame = ttk.Frame(self.notebook, style="TFrame")
        self.face_manage_frame = ttk.Frame(self.notebook, style="TFrame")
        self.log_frame = ttk.Frame(self.notebook, style="TFrame")  # 新增日志标签页
        
        self.notebook.add(self.encrypt_frame, text="文件加密")
        self.notebook.add(self.decrypt_frame, text="文件解密")
        self.notebook.add(self.face_manage_frame, text="人脸数据管理")
        self.notebook.add(self.log_frame, text="操作日志")  # 添加日志标签页
        
        # 初始化各标签页
        self.init_encrypt_tab()
        self.init_decrypt_tab()
        self.init_face_manage_tab()
        self.init_log_tab()  # 初始化日志标签页
        
        # 状态栏
        status_bar = tk.Frame(self.root, bg=self.COLORS['primary'], height=30)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        self.status_var = tk.StringVar()
        self.status_var.set(f"就绪 - 已加载 {len(self.known_names)} 个人脸数据")
        status_label = tk.Label(status_bar, 
                              textvariable=self.status_var,
                              bg=self.COLORS['primary'],
                              fg='white',
                              font=('Microsoft YaHei', 10),
                              anchor=tk.W)
        status_label.pack(side=tk.LEFT, padx=15)

    def init_encrypt_tab(self):
        """加密标签页"""
        main_frame = ttk.Frame(self.encrypt_frame)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        # 标题
        title_label = ttk.Label(main_frame, 
                              text="支持人脸加密和生成特征密钥", 
                              font=('Microsoft YaHei', 13, 'bold'),
                              foreground=self.COLORS['primary'])
        title_label.pack(anchor=tk.W, pady=(0, 15))
        
        # 人脸选择区域
        face_frame = ttk.LabelFrame(main_frame, text="人脸验证设置")
        face_frame.pack(fill=tk.X, pady=(0, 15))
        
        face_control_frame = ttk.Frame(face_frame)
        face_control_frame.pack(padx=15, pady=12, fill=tk.X)
        
        # 标签+下拉框组合
        select_frame = ttk.Frame(face_control_frame)
        select_frame.pack(side=tk.LEFT, padx=(0, 25))
        ttk.Label(select_frame, text="已录入人脸:", font=('Microsoft YaHei', 11)).pack(side=tk.LEFT, padx=(0, 8))
        self.face_combobox = ttk.Combobox(select_frame, width=28, state="readonly")
        self.face_combobox['values'] = self.known_names if self.known_names else ["无已录入人脸"]
        self.face_combobox.current(0) if self.known_names else self.face_combobox.set("无已录入人脸")
        self.face_combobox.pack(side=tk.LEFT)
        
        # 按钮组
        btn_frame = ttk.Frame(face_control_frame)
        btn_frame.pack(side=tk.LEFT)
        ttk.Button(btn_frame, text="录入新人脸", command=self.enroll_new_face, style="Primary.TButton").pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="选择当前人脸", command=self.select_current_face, style="Secondary.TButton").pack(side=tk.LEFT, padx=5)
        
        # 选中状态提示
        self.selected_face_label = ttk.Label(face_control_frame, 
                                           text="未选择人脸", 
                                           foreground=self.COLORS['danger'],
                                           font=('Microsoft YaHei', 11, 'bold'))
        self.selected_face_label.pack(side=tk.LEFT, padx=(30, 0))
        
        # 文件列表区域
        file_frame = ttk.LabelFrame(main_frame, text="待加密文件/文件夹")
        file_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 15))
        
        list_frame = ttk.Frame(file_frame)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=12)
        
        # 滚动条+列表框
        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.encrypt_listbox = tk.Listbox(list_frame, 
                                        yscrollcommand=scrollbar.set, 
                                        selectmode=tk.EXTENDED,
                                        bg=self.COLORS['card_bg'],
                                        fg=self.COLORS['text_primary'],
                                        borderwidth=1,
                                        relief=tk.FLAT,
                                        highlightbackground=self.COLORS['border'],
                                        highlightthickness=1,
                                        font=('Microsoft YaHei', 10),
                                        selectbackground=self.COLORS['secondary'],
                                        selectforeground='white')
        self.encrypt_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.encrypt_listbox.yview)
        
        # 文件操作按钮
        btn_frame = ttk.Frame(file_frame)
        btn_frame.pack(fill=tk.X, padx=15, pady=(0, 12))
        
        ttk.Button(btn_frame, text="添加文件", command=self.add_encrypt_files, style="Secondary.TButton").pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="选择文件夹", command=self.select_encrypt_folder, style="Secondary.TButton").pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="移除选中", command=self.remove_encrypt_selected, style="Secondary.TButton").pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="清空列表", command=self.clear_encrypt_list, style="Secondary.TButton").pack(side=tk.LEFT, padx=5)
        
        # 文件夹选择状态显示
        self.folder_status_label = ttk.Label(file_frame, 
                                            text="未选择加密文件夹", 
                                            foreground=self.COLORS['text_secondary'],
                                            font=('Microsoft YaHei', 10))
        self.folder_status_label.pack(anchor=tk.W, padx=15, pady=(0, 8))
        
        # 加密控制区域
        process_frame = ttk.Frame(main_frame)
        process_frame.pack(fill=tk.X, pady=10)
        
        self.encrypt_progress = ttk.Progressbar(process_frame, 
                                              orient=tk.HORIZONTAL, 
                                              length=100, 
                                              mode='determinate',
                                              style="Custom.Horizontal.TProgressbar")
        self.encrypt_progress.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 15))
        
        self.gen_feat_key_btn = ttk.Button(process_frame, text="生成特征密钥", command=self.generate_feat_key, state=tk.DISABLED, style="Secondary.TButton")
        self.gen_feat_key_btn.pack(side=tk.RIGHT, padx=5)
        
        self.encrypt_start_btn = ttk.Button(process_frame, text="开始加密", command=self.start_encryption, state=tk.DISABLED, style="Primary.TButton")
        self.encrypt_start_btn.pack(side=tk.RIGHT)
        
        # 说明文本
        desc_label = ttk.Label(main_frame, 
                             text="说明: 加密后可生成.feat密钥文件（含有效期），用于解密", 
                             foreground=self.COLORS['text_secondary'],
                             font=('Microsoft YaHei', 10))
        desc_label.pack(anchor=tk.W, pady=(5, 0))

    def init_decrypt_tab(self):
        """解密标签页"""
        main_frame = ttk.Frame(self.decrypt_frame)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        # 标题
        title_label = ttk.Label(main_frame, 
                              text="支持人脸/特征密钥解密", 
                              font=('Microsoft YaHei', 13, 'bold'),
                              foreground=self.COLORS['primary'])
        title_label.pack(anchor=tk.W, pady=(0, 15))
        
        # 解密方式选择
        decrypt_mode_frame = ttk.LabelFrame(main_frame, text="解密方式选择")
        decrypt_mode_frame.pack(fill=tk.X, pady=(0, 15))
        
        mode_control_frame = ttk.Frame(decrypt_mode_frame)
        mode_control_frame.pack(padx=15, pady=12, fill=tk.X)
        
        # 单选按钮
        self.decrypt_mode = tk.IntVar(value=0)
        radio1 = ttk.Radiobutton(mode_control_frame, 
                                text="人脸验证解密", 
                                variable=self.decrypt_mode, 
                                value=0, 
                                command=self.update_decrypt_ui)
        radio1.pack(side=tk.LEFT, padx=(0, 30))
        radio2 = ttk.Radiobutton(mode_control_frame, 
                                text="使用特征密钥解密", 
                                variable=self.decrypt_mode, 
                                value=1, 
                                command=self.update_decrypt_ui)
        radio2.pack(side=tk.LEFT)
        
        # 特征密钥文件选择
        self.feat_key_frame = ttk.Frame(decrypt_mode_frame)
        feat_select_frame = ttk.Frame(self.feat_key_frame)
        feat_select_frame.pack(padx=15, pady=8, fill=tk.X)
        
        ttk.Label(feat_select_frame, text="特征密钥文件:", font=('Microsoft YaHei', 11)).pack(side=tk.LEFT, padx=(0, 10))
        self.feat_key_path_var = tk.StringVar(value="未选择密钥文件")
        
        # 密钥路径显示标签
        path_label = ttk.Label(feat_select_frame, 
                             textvariable=self.feat_key_path_var,
                             foreground=self.COLORS['text_primary'],
                             background=self.COLORS['card_bg'],
                             borderwidth=1,
                             relief=tk.FLAT,
                             padding=6)
        path_label.pack(side=tk.LEFT, padx=(0, 15), fill=tk.X, expand=True)
        
        ttk.Button(feat_select_frame, text="选择密钥", command=self.select_feat_key, style="Secondary.TButton").pack(side=tk.LEFT)
        
        # 文件列表区域
        file_frame = ttk.LabelFrame(main_frame, text="待解密文件")
        file_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 15))
        
        list_frame = ttk.Frame(file_frame)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=12)
        
        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.decrypt_listbox = tk.Listbox(list_frame, 
                                        yscrollcommand=scrollbar.set, 
                                        selectmode=tk.EXTENDED,
                                        bg=self.COLORS['card_bg'],
                                        fg=self.COLORS['text_primary'],
                                        borderwidth=1,
                                        relief=tk.FLAT,
                                        highlightbackground=self.COLORS['border'],
                                        highlightthickness=1,
                                        font=('Microsoft YaHei', 10),
                                        selectbackground=self.COLORS['secondary'],
                                        selectforeground='white')
        self.decrypt_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.decrypt_listbox.yview)
        
        # 右键菜单绑定
        self.decrypt_listbox.bind("<Button-3>", self.show_decrypt_right_click_menu)
        self.decrypt_right_menu = tk.Menu(self.decrypt_listbox, tearoff=0)
        self.decrypt_right_menu.add_command(label="查询关联信息", command=self.query_encrypted_file_info)
        
        # 文件操作按钮
        btn_frame = ttk.Frame(file_frame)
        btn_frame.pack(fill=tk.X, padx=15, pady=(0, 12))
        
        ttk.Button(btn_frame, text="添加文件", command=self.add_decrypt_files, style="Secondary.TButton").pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="移除选中", command=self.remove_decrypt_selected, style="Secondary.TButton").pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="清空列表", command=self.clear_decrypt_list, style="Secondary.TButton").pack(side=tk.LEFT, padx=5)
        
        # 解密控制区域
        process_frame = ttk.Frame(main_frame)
        process_frame.pack(fill=tk.X, pady=10)
        
        self.decrypt_progress = ttk.Progressbar(process_frame, 
                                              orient=tk.HORIZONTAL, 
                                              length=100, 
                                              mode='determinate',
                                              style="Custom.Horizontal.TProgressbar")
        self.decrypt_progress.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 15))
        
        self.decrypt_start_btn = ttk.Button(process_frame, text="开始解密", command=self.start_decryption, style="Primary.TButton")
        self.decrypt_start_btn.pack(side=tk.RIGHT)
        
        # 说明文本
        desc_label = ttk.Label(main_frame, 
                             text="说明: 特征密钥文件(.feat)含有效期，过期将无法使用", 
                             foreground=self.COLORS['text_secondary'],
                             font=('Microsoft YaHei', 10))
        desc_label.pack(anchor=tk.W, pady=(5, 0))
        
        self.update_decrypt_ui()

    def init_face_manage_tab(self):
        """人脸数据管理页面"""
        main_frame = ttk.Frame(self.face_manage_frame)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        # 标题
        title_label = ttk.Label(main_frame, 
                              text="支持录入/删除人脸数据", 
                              font=('Microsoft YaHei', 13, 'bold'),
                              foreground=self.COLORS['primary'])
        title_label.pack(anchor=tk.W, pady=(0, 15))
        
        # 人脸列表
        list_frame = ttk.LabelFrame(main_frame, text="已录入人脸详情")
        list_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 15))
        
        # 创建表格视图
        columns = ("name", "enroll_time")
        self.face_tree = ttk.Treeview(list_frame, 
                                     columns=columns, 
                                     show="headings",
                                     style="Custom.Treeview")
        self.face_tree.heading("name", text="人脸名称")
        self.face_tree.heading("enroll_time", text="录入时间")
        self.face_tree.column("name", width=220, anchor=tk.CENTER)
        self.face_tree.column("enroll_time", width=320, anchor=tk.CENTER)
        
        # 添加滚动条
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.face_tree.yview)
        self.face_tree.configure(yscroll=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.face_tree.pack(fill=tk.BOTH, expand=True, padx=15, pady=12)
        
        # 加载人脸数据到表格
        self.refresh_face_tree()
        
        # 操作按钮
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill=tk.X, pady=10)
        
        # 左侧按钮
        left_btn_frame = ttk.Frame(btn_frame)
        left_btn_frame.pack(side=tk.LEFT)
        ttk.Button(left_btn_frame, text="刷新列表", command=self.refresh_face_tree, style="Secondary.TButton").pack(side=tk.LEFT, padx=5)
        ttk.Button(left_btn_frame, text="录入新人脸", command=self.enroll_new_face, style="Primary.TButton").pack(side=tk.LEFT, padx=5)
        
        # 右侧按钮
        right_btn_frame = ttk.Frame(btn_frame)
        right_btn_frame.pack(side=tk.RIGHT)
        ttk.Button(right_btn_frame, text="删除选中人脸", command=self.delete_selected_face, style="Danger.TButton").pack(side=tk.RIGHT, padx=5)
        
        # 警告提示
        warn_label = ttk.Label(main_frame, 
                             text="⚠️  提示：删除人脸数据后，所有关联的加密文件将无法解密，请谨慎操作！", 
                             foreground=self.COLORS['danger'],
                             font=('Microsoft YaHei', 10, 'bold'))
        warn_label.pack(anchor=tk.W, pady=(5, 0))

    def init_log_tab(self):
        """日志标签页初始化"""
        main_frame = ttk.Frame(self.log_frame)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        # 标题
        title_label = ttk.Label(main_frame, 
                              text="支持导出日志文件", 
                              font=('Microsoft YaHei', 13, 'bold'),
                              foreground=self.COLORS['primary'])
        title_label.pack(anchor=tk.W, pady=(0, 15))
        
        # 筛选与导出区域
        filter_frame = ttk.LabelFrame(main_frame, text="日志筛选与导出")
        filter_frame.pack(fill=tk.X, pady=(0, 15))
        
        filter_control = ttk.Frame(filter_frame)
        filter_control.pack(padx=15, pady=12, fill=tk.X)
        
        # 时间筛选（可选）
        ttk.Label(filter_control, text="时间范围：", font=('Microsoft YaHei', 11)).pack(side=tk.LEFT, padx=(0, 10))
        self.start_time_var = tk.StringVar(value="")
        self.end_time_var = tk.StringVar(value="")
        
        # 开始时间输入框
        start_entry = ttk.Entry(filter_control, textvariable=self.start_time_var, width=20)
        start_entry.pack(side=tk.LEFT, padx=(0, 10))
        start_entry.insert(0, "格式：2025-01-01 12:00:00")
        
        # 结束时间输入框
        end_entry = ttk.Entry(filter_control, textvariable=self.end_time_var, width=20)
        end_entry.pack(side=tk.LEFT, padx=(0, 20))
        end_entry.insert(0, "格式：2025-01-02 12:00:00")
        
        # 导出按钮
        ttk.Button(filter_control, text="导出日志", command=self.export_logs_click, style="Primary.TButton").pack(side=tk.LEFT)
        
        # 日志列表区域
        log_list_frame = ttk.LabelFrame(main_frame, text="日志记录")
        log_list_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 15))
        
        # 日志表格
        columns = ("time", "type", "content", "related", "status", "remark")
        self.log_tree = ttk.Treeview(log_list_frame, columns=columns, show="headings", style="Custom.Treeview")
        self.log_tree.heading("time", text="操作时间")
        self.log_tree.heading("type", text="操作类型")
        self.log_tree.heading("content", text="操作内容")
        self.log_tree.heading("related", text="关联对象")
        self.log_tree.heading("status", text="操作状态")
        self.log_tree.heading("remark", text="备注")
        
        # 列宽设置
        self.log_tree.column("time", width=200, anchor=tk.CENTER)
        self.log_tree.column("type", width=120, anchor=tk.CENTER)
        self.log_tree.column("content", width=150, anchor=tk.CENTER)
        self.log_tree.column("related", width=180, anchor=tk.CENTER)
        self.log_tree.column("status", width=80, anchor=tk.CENTER)
        self.log_tree.column("remark", width=250, anchor=tk.CENTER)
        
        # 滚动条
        scrollbar = ttk.Scrollbar(log_list_frame, orient=tk.VERTICAL, command=self.log_tree.yview)
        self.log_tree.configure(yscroll=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_tree.pack(fill=tk.BOTH, expand=True, padx=15, pady=12)
        
        # 刷新日志按钮
        ttk.Button(main_frame, text="刷新日志列表", command=self.refresh_log_list, style="Secondary.TButton").pack(anchor=tk.W)
        
        # 初始刷新日志
        self.refresh_log_list()

    # ---------------------- 核心功能方法 ----------------------
    def refresh_face_tree(self):
        """刷新人脸列表表格数据"""
        for item in self.face_tree.get_children():
            self.face_tree.delete(item)
        for name, timestamp in zip(self.known_names, self.enroll_timestamps):
            enroll_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))
            self.face_tree.insert("", tk.END, values=(name, enroll_time))

    def delete_selected_face(self):
        """删除选中的人脸数据"""
        selected_items = self.face_tree.selection()
        if not selected_items:
            messagebox.showwarning("提示", "请先选择要删除的人脸数据")
            return
        
        selected_names = [self.face_tree.item(item, "values")[0] for item in selected_items]
        
        if not messagebox.askyesno("确认删除", 
            f"确定要删除以下人脸数据吗？\n{', '.join(selected_names)}\n删除后将无法解密关联文件！"):
            return
        
        try:
            for name in selected_names:
                idx = self.known_names.index(name)
                self.known_names.pop(idx)
                self.known_features = np.delete(self.known_features, idx, axis=0)
                self.enroll_timestamps.pop(idx)
            
            with open("insight_face_db.pkl", "wb") as f:
                pickle.dump((self.known_features.tolist(), self.known_names, self.enroll_timestamps), f)
            
            self.refresh_face_tree()
            self.face_combobox['values'] = self.known_names if self.known_names else ["无已录入人脸"]
            self.face_combobox.current(0) if self.known_names else self.face_combobox.set("无已录入人脸")
            
            if self.selected_face_label.cget("text").replace("已选择：", "") in selected_names:
                self.selected_face_feat = None
                self.selected_face_label.config(text="未选择人脸", foreground=self.COLORS['danger'])
                self.encrypt_start_btn.config(state=tk.DISABLED)
                self.gen_feat_key_btn.config(state=tk.DISABLED)
            
            self.status_var.set(f"已删除 {len(selected_names)} 个人脸数据")
            messagebox.showinfo("成功", f"已成功删除 {len(selected_names)} 个人脸数据")
            
            # 写入日志
            write_log(
                operation_type="人脸管理",
                operation_content="删除人脸数据",
                related_obj=", ".join(selected_names),
                status="成功",
                remark="手动删除"
            )
            
        except Exception as e:
            err_msg = str(e)
            messagebox.showerror("错误", f"删除失败：{err_msg}")
            self.status_var.set(f"删除人脸数据失败：{err_msg}")
            write_log(
                operation_type="人脸管理",
                operation_content="删除人脸数据",
                related_obj=", ".join(selected_names),
                status="失败",
                remark=err_msg
            )

    def update_decrypt_ui(self):
        if self.decrypt_mode.get() == 1:
            self.feat_key_frame.pack(fill=tk.X, padx=15, pady=8)
        else:
            self.feat_key_frame.pack_forget()

    def select_feat_key(self):
        feat_path = filedialog.askopenfilename(
            title="选择特征密钥文件",
            filetypes=[("特征密钥文件", "*.feat")]
        )
        if feat_path:
            self.tmp_key_path = feat_path
            self.feat_key_path_var.set(os.path.basename(feat_path))
            self.status_var.set(f"已选择特征密钥文件: {os.path.basename(feat_path)}")
            write_log(
                operation_type="密钥管理",
                operation_content="选择特征密钥文件",
                related_obj=os.path.basename(feat_path),
                status="成功",
                remark=f"路径：{feat_path}"
            )

    def enroll_new_face(self):
        name = simpledialog.askstring("录入新人脸", "请输入人脸名称（尽量为英文）:")
        if not name or name.strip() == "":
            messagebox.showwarning("警告", "人脸名称不能为空！")
            return
        
        is_override = name in self.known_names
        if is_override and not messagebox.askyesno("确认", "该名称已存在，是否覆盖？"):
            return
        
        def enroll_callback(feat, timestamp):
            if feat is not None:
                if name in self.known_names:
                    idx = self.known_names.index(name)
                    self.known_features[idx] = feat
                    self.enroll_timestamps[idx] = timestamp
                else:
                    self.known_features = np.append(self.known_features, [feat], axis=0) if len(self.known_features) > 0 else np.array([feat])
                    self.known_names.append(name)
                    self.enroll_timestamps.append(timestamp)
                
                with open("insight_face_db.pkl", "wb") as f:
                    pickle.dump((self.known_features.tolist(), self.known_names, self.enroll_timestamps), f)
                
                self.face_combobox['values'] = self.known_names
                self.face_combobox.current(len(self.known_names)-1)
                self.refresh_face_tree()
                self.status_var.set(f"已录入新人脸：{name}")
                
                # 写入日志
                write_log(
                    operation_type="人脸管理",
                    operation_content="录入新人脸" if not is_override else "覆盖人脸数据",
                    related_obj=name,
                    status="成功",
                    remark=f"时间戳：{timestamp}"
                )
        
        threading.Thread(target=enroll_face_gui, args=(name, enroll_callback), daemon=True).start()

    def select_current_face(self):
        selected_name = self.face_combobox.get()
        if selected_name == "无已录入人脸":
            messagebox.showwarning("警告", "暂无已录入人脸，请先录入！")
            return
        
        idx = self.known_names.index(selected_name)
        self.selected_face_feat = self.known_features[idx]
        
        self.selected_face_label.config(text=f"已选择：{selected_name}", foreground=self.COLORS['success'])
        self.encrypt_start_btn.config(state=tk.NORMAL)
        self.gen_feat_key_btn.config(state=tk.NORMAL)
        self.status_var.set(f"已选择加密关联人脸：{selected_name}")
        write_log(
            operation_type="人脸管理",
            operation_content="选择加密关联人脸",
            related_obj=selected_name,
            status="成功",
            remark="用于文件加密"
        )

    # 文件操作方法
    def add_encrypt_files(self):
        files = filedialog.askopenfilenames(title="选择要加密的文件")
        if files:
            added_count = 0
            for file in files:
                if file not in self.encrypt_files:
                    self.encrypt_files.append(file)
                    self.encrypt_listbox.insert(tk.END, os.path.basename(file))
                    added_count += 1
            self.status_var.set(f"已添加 {added_count} 个文件到加密列表")
            write_log(
                operation_type="文件管理",
                operation_content="添加加密文件",
                related_obj=f"共{added_count}个文件",
                status="成功",
                remark=f"总加密文件数：{len(self.encrypt_files)}"
            )

    def select_encrypt_folder(self):
        """选择待加密的文件夹"""
        folder_path = filedialog.askdirectory(title="选择要加密的文件夹")
        if not folder_path:
            return
        
        # 获取文件夹内所有文件
        self.folder_files = get_all_files_in_folder(folder_path)
        if not self.folder_files:
            messagebox.showwarning("提示", "所选文件夹内无有效文件（已跳过隐藏文件）")
            return
        
        # 生成默认保存路径（原文件夹同级，添加时间戳）
        folder_name = os.path.basename(folder_path)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        default_save_path = os.path.join(os.path.dirname(folder_path), f"{folder_name}_加密_{timestamp}")
        
        # 让用户确认保存路径
        save_path = filedialog.askdirectory(title="选择加密后文件夹的保存位置", initialdir=default_save_path)
        if not save_path:
            return
        
        self.selected_folder = folder_path
        self.encrypt_folder_save_path = save_path
        
        # 更新状态显示
        file_count = len(self.folder_files)
        self.folder_status_label.config(
            text=f"已选择文件夹：{folder_name}（含 {file_count} 个文件），加密后保存至：{os.path.basename(save_path)}",
            foreground=self.COLORS['success']
        )
        self.status_var.set(f"已选择加密文件夹：{folder_path}（{file_count} 个文件）")
        
        # 自动启用加密按钮（如果已选择人脸）
        if self.selected_face_feat is not None:
            self.encrypt_start_btn.config(state=tk.NORMAL)
        
        # 写入日志
        write_log(
            operation_type="文件管理",
            operation_content="选择加密文件夹",
            related_obj=folder_name,
            status="成功",
            remark=f"包含{file_count}个文件，保存至：{save_path}"
        )

    def remove_encrypt_selected(self):
        # 清除文件夹选择状态
        prev_folder = self.selected_folder
        self.selected_folder = None
        self.encrypt_folder_save_path = None
        self.folder_files = []
        self.folder_status_label.config(text="未选择加密文件夹", foreground=self.COLORS['text_secondary'])
        
        # 清除文件列表
        selected = self.encrypt_listbox.curselection()
        if not selected:
            return
        removed_count = len(selected)
        for i in sorted(selected, reverse=True):
            self.encrypt_listbox.delete(i)
            del self.encrypt_files[i]
        
        # 写入日志
        log_content = "移除加密文件夹" if prev_folder else "移除加密文件"
        log_related = os.path.basename(prev_folder) if prev_folder else f"共{removed_count}个文件"
        self.status_var.set(f"已移除 {removed_count} 个文件/文件夹")
        write_log(
            operation_type="文件管理",
            operation_content=log_content,
            related_obj=log_related,
            status="成功",
            remark=""
        )

    def clear_encrypt_list(self):
        if messagebox.askyesno("确认", "确定要清空加密文件和文件夹列表吗？"):
            # 记录操作前状态
            prev_file_count = len(self.encrypt_files)
            prev_has_folder = bool(self.selected_folder)
            
            # 清除文件列表
            self.encrypt_listbox.delete(0, tk.END)
            self.encrypt_files = []
            
            # 清除文件夹选择状态
            self.selected_folder = None
            self.encrypt_folder_save_path = None
            self.folder_files = []
            self.folder_status_label.config(text="未选择加密文件夹", foreground=self.COLORS['text_secondary'])
            
            self.status_var.set("加密文件和文件夹列表已清空")
            
            # 写入日志
            related = f"{prev_file_count}个文件+1个文件夹" if prev_has_folder else f"{prev_file_count}个文件"
            write_log(
                operation_type="文件管理",
                operation_content="清空加密列表",
                related_obj=related,
                status="成功",
                remark="手动清空"
            )

    def add_decrypt_files(self):
        files = filedialog.askopenfilenames(title="选择要解密的文件", filetypes=[("加密文件", "*.enc")])
        if files:
            added_count = 0
            for file in files:
                if file not in self.decrypt_files:
                    self.decrypt_files.append(file)
                    self.decrypt_listbox.insert(tk.END, os.path.basename(file))
                    added_count += 1
            self.status_var.set(f"已添加 {added_count} 个文件到解密列表")
            write_log(
                operation_type="文件管理",
                operation_content="添加解密文件",
                related_obj=f"共{added_count}个文件",
                status="成功",
                remark=f"总解密文件数：{len(self.decrypt_files)}"
            )

    def remove_decrypt_selected(self):
        selected = self.decrypt_listbox.curselection()
        if not selected:
            return
        removed_count = len(selected)
        for i in sorted(selected, reverse=True):
            self.decrypt_listbox.delete(i)
            del self.decrypt_files[i]
        self.status_var.set(f"已移除 {removed_count} 个文件")
        write_log(
            operation_type="文件管理",
            operation_content="移除解密文件",
            related_obj=f"共{removed_count}个文件",
            status="成功",
            remark=f"剩余解密文件数：{len(self.decrypt_files)}"
        )

    def clear_decrypt_list(self):
        if messagebox.askyesno("确认", "确定要清空解密文件列表吗？"):
            prev_count = len(self.decrypt_files)
            self.decrypt_listbox.delete(0, tk.END)
            self.decrypt_files = []
            self.status_var.set("解密文件列表已清空")
            write_log(
                operation_type="文件管理",
                operation_content="清空解密列表",
                related_obj=f"共{prev_count}个文件",
                status="成功",
                remark="手动清空"
            )

    # 特征密钥生成功能
    def generate_feat_key(self):
        if self.selected_face_feat is None:
            messagebox.showwarning("警告", "请先选择人脸！")
            return
        
        valid_days = simpledialog.askinteger(
            "设置密钥有效期", 
            "请输入密钥有效天数（1-30天）：",
            minvalue=1,
            maxvalue=30,
            initialvalue=7
        )
        if valid_days is None:
            return
        
        save_path = filedialog.asksaveasfilename(
            title="保存特征密钥文件",
            defaultextension=".feat",
            filetypes=[("特征密钥文件", "*.feat")]
        )
        if not save_path:
            return
        
        try:
            feat_str = feat_to_str(self.selected_face_feat)
            create_timestamp = int(time.time())
            key_data = f"{feat_str}|{create_timestamp}|{valid_days}"
            
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(key_data)
            
            expire_time = time.strftime(
                "%Y-%m-%d %H:%M:%S",
                time.localtime(create_timestamp + valid_days * 86400)
            )
            self.status_var.set(f"特征密钥生成成功：{os.path.basename(save_path)}（有效期至{expire_time}）")
            messagebox.showinfo(
                "成功", 
                f"特征密钥已保存至：\n{save_path}\n有效期：{valid_days}天\n过期时间：{expire_time}"
            )
            
            # 写入日志
            selected_name = self.face_combobox.get()
            write_log(
                operation_type="密钥管理",
                operation_content="生成特征密钥",
                related_obj=os.path.basename(save_path),
                status="成功",
                remark=f"关联人脸：{selected_name}，有效期{valid_days}天，过期时间{expire_time}"
            )
            
        except Exception as e:
            err_msg = str(e)
            messagebox.showerror("错误", f"生成特征密钥失败：{err_msg}")
            self.status_var.set(f"生成特征密钥失败：{err_msg}")
            write_log(
                operation_type="密钥管理",
                operation_content="生成特征密钥",
                related_obj="未知",
                status="失败",
                remark=err_msg
            )

    # 加密解密核心逻辑
    def get_encryption_key(self, face_feat, salt=None):
        if salt is None:
            salt = get_random_bytes(16)
        feat_bytes = face_feat.tobytes()
        key = PBKDF2(feat_bytes, salt, dkLen=32, count=100000)
        return key, salt

    def start_encryption(self):
        if self.is_processing or self.selected_face_feat is None:
            return
        
        # 支持文件列表和文件夹两种模式
        has_files = len(self.encrypt_files) > 0
        has_folder = len(self.folder_files) > 0
        
        if not has_files and not has_folder:
            messagebox.showerror("错误", "请添加要加密的文件或选择加密文件夹")
            return
        
        self.encrypt_start_btn.config(state=tk.DISABLED)
        self.is_processing = True
        
        if has_folder:
            self.status_var.set(f"开始加密文件夹：{os.path.basename(self.selected_folder)}...")
            threading.Thread(target=self.perform_folder_encryption, daemon=True).start()
        else:
            self.status_var.set("开始加密文件...")
            threading.Thread(target=self.perform_encryption, daemon=True).start()

    def perform_encryption(self):
        total = len(self.encrypt_files)
        success = 0
        selected_name = self.face_combobox.get()
        encrypt_algorithm = "AES-GCM-256"
        
        for i, file_path in enumerate(self.encrypt_files):
            try:
                with open(file_path, 'rb') as f:
                    data = f.read()
                
                key, salt = self.get_encryption_key(self.selected_face_feat)
                cipher = AES.new(key, AES.MODE_GCM)
                ciphertext, tag = cipher.encrypt_and_digest(data)
                
                # 编码关联信息
                name_bytes = selected_name.encode('utf-8')
                name_len = len(name_bytes).to_bytes(4, byteorder='little')  # 4字节存储名称长度
                encrypt_timestamp = int(time.time())
                timestamp_bytes = encrypt_timestamp.to_bytes(8, byteorder='little')  # 8字节存储时间戳
                algorithm_bytes = encrypt_algorithm.ljust(16, '\x00').encode('utf-8')  # 16字节存储算法（填充）
                key_valid_days = 0  # 默认为永久
                
                # 写入加密文件
                encrypted_file_path = file_path + ".enc"
                with open(encrypted_file_path, 'wb') as f:
                    f.write(salt)
                    f.write(cipher.nonce)
                    f.write(tag)
                    f.write(self.selected_face_feat.tobytes())
                    # 新增关联信息
                    f.write(name_len)
                    f.write(name_bytes)
                    f.write(timestamp_bytes)
                    f.write(algorithm_bytes)
                    f.write(key_valid_days.to_bytes(4, byteorder='little'))
                    # 原有密文
                    f.write(ciphertext)
                
                success += 1
                self.status_var.set(f"正在加密: {os.path.basename(file_path)} ({i+1}/{total})")
                # 写入日志
                write_log(
                    operation_type="文件加密",
                    operation_content="加密单个文件",
                    related_obj=os.path.basename(file_path),
                    status="成功",
                    remark=f"关联人脸：{selected_name}，算法：{encrypt_algorithm}"
                )
                
            except Exception as e:
                err_msg = str(e)
                self.status_var.set(f"加密失败 {os.path.basename(file_path)}: {err_msg}")
                # 写入失败日志
                write_log(
                    operation_type="文件加密",
                    operation_content="加密单个文件",
                    related_obj=os.path.basename(file_path),
                    status="失败",
                    remark=err_msg
                )
            
            self.root.after(10, lambda val=(i+1)/total*100: self.encrypt_progress.configure(value=val))
        
        # 加密完成日志
        write_log(
            operation_type="文件加密",
            operation_content="批量加密文件",
            related_obj=f"共{total}个文件",
            status="完成",
            remark=f"成功{success}个，失败{total-success}个"
        )
        
        self.root.after(10, lambda: self.encrypt_start_btn.config(state=tk.NORMAL))
        self.root.after(10, lambda: self.encrypt_progress.configure(value=0))
        self.is_processing = False
        self.root.after(10, lambda: self.status_var.set(f"加密完成: 成功 {success} 个，失败 {total - success} 个"))
        self.root.after(10, lambda: messagebox.showinfo("完成", f"加密完成: 成功 {success} 个，失败 {total - success} 个"))

    def perform_folder_encryption(self):
        """执行文件夹加密（保持原结构）"""
        total = len(self.folder_files)
        success = 0
        selected_name = self.face_combobox.get()
        encrypt_algorithm = "AES-GCM-256"
        folder_name = os.path.basename(self.selected_folder)
        
        for i, (file_abs_path, file_rel_path) in enumerate(self.folder_files):
            try:
                # 创建加密后的文件夹结构
                encrypt_rel_dir = os.path.dirname(file_rel_path)
                encrypt_abs_dir = os.path.join(self.encrypt_folder_save_path, encrypt_rel_dir)
                create_folder_if_not_exists(encrypt_abs_dir)
                
                # 读取原文件
                with open(file_abs_path, 'rb') as f:
                    data = f.read()
                
                key, salt = self.get_encryption_key(self.selected_face_feat)
                cipher = AES.new(key, AES.MODE_GCM)
                ciphertext, tag = cipher.encrypt_and_digest(data)
                
                # 编码关联信息
                name_bytes = selected_name.encode('utf-8')
                name_len = len(name_bytes).to_bytes(4, byteorder='little')
                encrypt_timestamp = int(time.time())
                timestamp_bytes = encrypt_timestamp.to_bytes(8, byteorder='little')
                algorithm_bytes = encrypt_algorithm.ljust(16, '\x00').encode('utf-8')
                key_valid_days = 0  # 默认为永久
                
                # 生成加密后文件路径
                encrypt_file_name = f"{os.path.basename(file_rel_path)}.enc"
                encrypt_file_path = os.path.join(encrypt_abs_dir, encrypt_file_name)
                
                # 写入加密文件
                with open(encrypt_file_path, 'wb') as f:
                    f.write(salt)
                    f.write(cipher.nonce)
                    f.write(tag)
                    f.write(self.selected_face_feat.tobytes())
                    f.write(name_len)
                    f.write(name_bytes)
                    f.write(timestamp_bytes)
                    f.write(algorithm_bytes)
                    f.write(key_valid_days.to_bytes(4, byteorder='little'))
                    f.write(ciphertext)
                
                success += 1
                self.status_var.set(f"正在加密: {file_rel_path} ({i+1}/{total})")
                # 写入日志
                write_log(
                    operation_type="文件夹加密",
                    operation_content="加密文件夹内文件",
                    related_obj=file_rel_path,
                    status="成功",
                    remark=f"关联人脸：{selected_name}，算法：{encrypt_algorithm}"
                )
                
            except Exception as e:
                err_msg = str(e)
                self.status_var.set(f"加密失败 {file_rel_path}: {err_msg}")
                # 写入失败日志
                write_log(
                    operation_type="文件夹加密",
                    operation_content="加密文件夹内文件",
                    related_obj=file_rel_path,
                    status="失败",
                    remark=err_msg
                )
            
            self.root.after(10, lambda val=(i+1)/total*100: self.encrypt_progress.configure(value=val))
        
        # 文件夹加密完成日志
        write_log(
            operation_type="文件夹加密",
            operation_content="批量加密文件夹",
            related_obj=folder_name,
            status="完成",
            remark=f"共{total}个文件，成功{success}个，失败{total-success}个"
        )
        
        # 完成后清理文件夹选择状态
        self.selected_folder = None
        self.encrypt_folder_save_path = None
        self.folder_files = []
        self.root.after(10, lambda: self.folder_status_label.config(text="未选择加密文件夹", foreground=self.COLORS['text_secondary']))
        
        self.root.after(10, lambda: self.encrypt_start_btn.config(state=tk.NORMAL))
        self.root.after(10, lambda: self.encrypt_progress.configure(value=0))
        self.is_processing = False
        self.root.after(10, lambda: self.status_var.set(f"文件夹加密完成: 成功 {success} 个，失败 {total - success} 个"))
        self.root.after(10, lambda: messagebox.showinfo("完成", f"文件夹加密完成: 成功 {success} 个，失败 {total - success} 个\n加密文件保存至：{self.encrypt_folder_save_path}"))

    def start_decryption(self):
        if self.is_processing:
            return
        
        if len(self.decrypt_files) == 0:
            messagebox.showerror("错误", "请添加要解密的文件")
            return
        
        if self.decrypt_mode.get() == 1:
            if not self.tmp_key_path or not os.path.exists(self.tmp_key_path):
                messagebox.showerror("错误", "请先选择有效的特征密钥文件")
                return
            self.start_feat_key_decryption()
        else:
            self.start_face_verification_decryption()

    def start_face_verification_decryption(self):
        self.decrypt_start_btn.config(state=tk.DISABLED)
        self.is_processing = True
        self.status_var.set("开始人脸验证...（请查看弹出的验证窗口）")
        
        first_file = self.decrypt_files[0]
        try:
            with open(first_file, 'rb') as f:
                f.seek(16 + 16 + 16)
                target_feat_bytes = f.read(512 * 4)
                target_feat = np.frombuffer(target_feat_bytes, dtype=np.float32)
            
            def verify_callback(result):
                if result:
                    self.status_var.set("人脸验证通过，开始解密...")
                    threading.Thread(target=self.perform_decryption, args=(target_feat,), daemon=True).start()
                else:
                    self.status_var.set("人脸验证失败，解密终止")
                    self.root.after(10, lambda: self.decrypt_start_btn.config(state=tk.NORMAL))
                    self.is_processing = False
                    messagebox.showerror("失败", "人脸验证失败，无解密权限！")
                    write_log(
                        operation_type="文件解密",
                        operation_content="人脸验证解密",
                        related_obj=f"共{len(self.decrypt_files)}个文件",
                        status="失败",
                        remark="人脸验证未通过"
                    )
            
            threading.Thread(target=verify_face, args=(target_feat, verify_callback), daemon=True).start()
        
        except Exception as e:
            err_msg = str(e)
            messagebox.showerror("错误", f"读取加密文件失败: {err_msg}")
            self.status_var.set("解密失败: 读取文件错误")
            self.decrypt_start_btn.config(state=tk.NORMAL)
            self.is_processing = False
            write_log(
                operation_type="文件解密",
                operation_content="人脸验证解密",
                related_obj=os.path.basename(first_file),
                status="失败",
                remark=f"读取文件错误: {err_msg}"
            )

    def start_feat_key_decryption(self):
        self.decrypt_start_btn.config(state=tk.DISABLED)
        self.is_processing = True
        self.status_var.set("正在验证特征密钥...")
        
        try:
            with open(self.tmp_key_path, 'r', encoding='utf-8') as f:
                key_data = f.read().strip()
            
            parts = key_data.split('|')
            if len(parts) != 3:
                raise ValueError("密钥文件格式错误，可能是旧版无有效期的密钥")
            
            feat_str, create_timestamp_str, valid_days_str = parts
            create_timestamp = int(create_timestamp_str)
            valid_days = int(valid_days_str)
            
            current_timestamp = int(time.time())
            expire_timestamp = create_timestamp + valid_days * 86400
            
            if current_timestamp > expire_timestamp:
                expire_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(expire_timestamp))
                raise ValueError(f"特征密钥已过期！\n过期时间：{expire_time}\n请重新生成密钥")
            
            key_feat = str_to_feat(feat_str)
            
            first_file = self.decrypt_files[0]
            with open(first_file, 'rb') as f:
                f.seek(16 + 16 + 16)
                file_feat = np.frombuffer(f.read(512 * 4), dtype=np.float32)
            
            sim_score = np.dot(key_feat, file_feat)
            if sim_score < 0.85:
                raise ValueError(f"特征密钥与加密文件不匹配（相似度: {sim_score:.4f}）")
            
            remaining_days = (expire_timestamp - current_timestamp) / 86400
            self.status_var.set(f"特征密钥验证通过（剩余有效期：{remaining_days:.1f}天），开始解密...")
            threading.Thread(target=self.perform_decryption, args=(key_feat,), daemon=True).start()
            
        except ValueError as e:
            err_msg = str(e)
            messagebox.showerror("错误", f"特征密钥验证失败: {err_msg}")
            self.status_var.set(f"解密失败: {err_msg}")
            self.decrypt_start_btn.config(state=tk.NORMAL)
            self.is_processing = False
            write_log(
                operation_type="文件解密",
                operation_content="特征密钥解密",
                related_obj=os.path.basename(self.tmp_key_path),
                status="失败",
                remark=err_msg
            )
        except Exception as e:
            err_msg = str(e)
            messagebox.showerror("错误", f"特征密钥读取失败: {err_msg}")
            self.status_var.set(f"解密失败: {err_msg}")
            self.decrypt_start_btn.config(state=tk.NORMAL)
            self.is_processing = False
            write_log(
                operation_type="文件解密",
                operation_content="特征密钥解密",
                related_obj=os.path.basename(self.tmp_key_path) if self.tmp_key_path else "未知",
                status="失败",
                remark=err_msg
            )

    def perform_decryption(self, target_feat):
        total = len(self.decrypt_files)
        success = 0
        
        for i, file_path in enumerate(self.decrypt_files):
            try:
                with open(file_path, 'rb') as f:
                    salt = f.read(16)
                    nonce = f.read(16)
                    tag = f.read(16)
                    stored_feat_bytes = f.read(512 * 4)
                    # 跳过关联信息字段（不影响解密）
                    try:
                        name_len = int.from_bytes(f.read(4), byteorder='little')
                        f.read(name_len + 8 + 16 + 4)  # 跳过名称、时间戳、算法、有效期
                    except:
                        pass
                    ciphertext = f.read()
                
                stored_feat = np.frombuffer(stored_feat_bytes, dtype=np.float32)
                sim_score = np.dot(stored_feat, target_feat)
                if sim_score < 0.85:
                    err_msg = f"特征不匹配（相似度: {sim_score:.4f}）"
                    self.status_var.set(f"解密失败 {os.path.basename(file_path)}: {err_msg}")
                    write_log(
                        operation_type="文件解密",
                        operation_content="解密单个文件",
                        related_obj=os.path.basename(file_path),
                        status="失败",
                        remark=err_msg
                    )
                    continue
                
                key, _ = self.get_encryption_key(target_feat, salt)
                cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
                data = cipher.decrypt_and_verify(ciphertext, tag)
                
                decrypted_file_path = self.get_decrypted_path(file_path)
                with open(decrypted_file_path, 'wb') as f:
                    f.write(data)
                
                success += 1
                self.status_var.set(f"正在解密: {os.path.basename(file_path)} ({i+1}/{total})")
                write_log(
                    operation_type="文件解密",
                    operation_content="解密单个文件",
                    related_obj=os.path.basename(file_path),
                    status="成功",
                    remark=f"保存至：{os.path.basename(decrypted_file_path)}"
                )
                
            except ValueError as e:
                err_msg = "文件损坏或密钥错误"
                self.status_var.set(f"解密失败 {os.path.basename(file_path)}: {err_msg}")
                write_log(
                    operation_type="文件解密",
                    operation_content="解密单个文件",
                    related_obj=os.path.basename(file_path),
                    status="失败",
                    remark=err_msg
                )
            except Exception as e:
                err_msg = str(e)
                self.status_var.set(f"解密失败 {os.path.basename(file_path)}: {err_msg}")
                write_log(
                    operation_type="文件解密",
                    operation_content="解密单个文件",
                    related_obj=os.path.basename(file_path),
                    status="失败",
                    remark=err_msg
                )
            
            self.root.after(10, lambda val=(i+1)/total*100: self.decrypt_progress.configure(value=val))
        
        self.finish_decryption(success, total)

    def get_decrypted_path(self, file_path):
        if file_path.endswith(".enc"):
            decrypted_file_path = file_path[:-4]
        else:
            decrypted_file_path = file_path + ".decrypted"
        
        if os.path.exists(decrypted_file_path):
            base, ext = os.path.splitext(decrypted_file_path)
            counter = 1
            while os.path.exists(f"{base}_{counter}{ext}"):
                counter += 1
            decrypted_file_path = f"{base}_{counter}{ext}"
        return decrypted_file_path

    def finish_decryption(self, success, total):
        self.root.after(10, lambda: self.decrypt_start_btn.config(state=tk.NORMAL))
        self.root.after(10, lambda: self.decrypt_progress.configure(value=0))
        self.is_processing = False
        self.root.after(10, lambda: self.status_var.set(f"解密完成: 成功 {success} 个，失败 {total - success} 个"))
        self.root.after(10, lambda: messagebox.showinfo("完成", f"解密完成: 成功 {success} 个，失败 {total - success} 个"))
        
        # 写入日志
        write_log(
            operation_type="文件解密",
            operation_content="批量解密文件",
            related_obj=f"共{total}个文件",
            status="完成",
            remark=f"成功{success}个，失败{total-success}个"
        )

    # 新增：加密文件关联信息查询相关方法
    def show_decrypt_right_click_menu(self, event):
        """显示解密列表右键菜单"""
        try:
            # 选中点击的文件
            index = self.decrypt_listbox.nearest(event.y)
            self.decrypt_listbox.selection_clear(0, tk.END)
            self.decrypt_listbox.selection_set(index)
            # 显示菜单
            self.decrypt_right_menu.post(event.x_root, event.y_root)
        except:
            pass

    def query_encrypted_file_info(self):
        """查询选中加密文件的关联信息"""
        selected = self.decrypt_listbox.curselection()
        if not selected:
            messagebox.showwarning("提示", "请先选中要查询的加密文件")
            return
        
        # 获取选中文件路径
        selected_idx = selected[0]
        file_path = self.decrypt_files[selected_idx]
        file_name = os.path.basename(file_path)
        
        # 解析关联信息
        info = parse_encrypted_file_info(file_path, self.known_features, self.known_names)
        
        # 创建信息窗口
        info_window = tk.Toplevel(self.root)
        info_window.title(f"文件关联信息 - {file_name}")
        info_window.geometry("450x300")
        info_window.resizable(False, False)
        
        # 布局信息
        main_frame = ttk.Frame(info_window, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        if info["status"] == "error":
            ttk.Label(main_frame, text="解析失败", font=('Microsoft YaHei', 12, 'bold'), foreground=self.COLORS['danger']).pack(anchor=tk.W, pady=5)
            ttk.Label(main_frame, text=info["error_msg"], foreground=self.COLORS['text_primary']).pack(anchor=tk.W, pady=5)
            return
        
        # 显示信息（分字段）
        fields = [
            ("文件名称", file_name),
            ("关联人脸", info["face_name"]),
            ("加密时间", info["encrypt_time"]),
            ("加密算法", info["encrypt_algorithm"]),
            ("密钥有效期", info["key_valid_days"]),
            ("人脸相似度匹配", info["similarity"])
        ]
        
        for label_text, value_text in fields:
            frame = ttk.Frame(main_frame)
            frame.pack(fill=tk.X, pady=4)
            ttk.Label(frame, text=label_text, width=12, font=('Microsoft YaHei', 11, 'bold')).pack(side=tk.LEFT)
            ttk.Label(frame, text=value_text, foreground=self.COLORS['text_primary']).pack(side=tk.LEFT)
        
        # 旧版文件提示
        if info["is_old_version"]:
            ttk.Label(main_frame, text="⚠️  该文件为旧版加密，部分信息无法显示", foreground=self.COLORS['warning'], font=('Microsoft YaHei', 10)).pack(anchor=tk.W, pady=(10, 0))
        
        # 写入查询日志
        write_log(
            operation_type="文件信息查询",
            operation_content="查询加密文件关联信息",
            related_obj=file_name,
            status="成功",
            remark=f"文件路径：{file_path}"
        )

    # 新增：日志管理相关方法
    def refresh_log_list(self):
        """刷新日志列表"""
        # 清空现有数据
        for item in self.log_tree.get_children():
            self.log_tree.delete(item)
        
        # 读取日志文件
        if not os.path.exists(LOG_FILE_PATH):
            return
        
        with open(LOG_FILE_PATH, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)  # 跳过表头
            # 倒序显示（最新日志在顶部）
            logs = list(reader)[::-1]
            for log in logs:
                self.log_tree.insert("", tk.END, values=log)

    def export_logs_click(self):
        """导出日志按钮点击事件"""
        start_time = self.start_time_var.get().strip()
        end_time = self.end_time_var.get().strip()
        
        # 验证时间格式（可选）
        if start_time and start_time != "格式：2025-01-01 12:00:00":
            try:
                datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
            except:
                messagebox.showwarning("提示", "开始时间格式错误，请按示例填写")
                return
        else:
            start_time = None
        
        if end_time and end_time != "格式：2025-01-02 12:00:00":
            try:
                datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S")
            except:
                messagebox.showwarning("提示", "结束时间格式错误，请按示例填写")
                return
        else:
            end_time = None
        
        # 导出日志
        success, msg = export_logs(start_time, end_time)
        if success:
            messagebox.showinfo("成功", msg)
            # 写入导出日志
            write_log(
                operation_type="日志导出",
                operation_content="导出操作日志",
                related_obj=f"时间范围：{start_time or '无'} 至 {end_time or '无'}",
                status="成功",
                remark=msg
            )
        else:
            messagebox.showwarning("提示", msg)

# ---------------------- 运行入口 ----------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = FaceFileEncryptorApp(root)
    root.mainloop()