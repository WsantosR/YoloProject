
import cv2
from pathlib import Path
import time
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
from ultralytics import YOLO
import threading

PROJECT_DIR = Path.home() / "Projetos" / "YoloProject"

MODEL_PATH = (
    PROJECT_DIR / "runs" / "detect" / "yolov10m"
    / "weights" / "best.pt"
)  # Caminho do modelo treinado
CAMERA_ID = 0           # 0 = câmera padrão
CONFIDENCE = 0.50       # Confiança mínima


class YOLOInterface:
    def __init__(self, root):
        self.root = root
        self.root.title("Sistema de Detecção YOLO")
        self.root.geometry("1100x750")
        self.root.configure(bg="#171923")

        # Carrega o modelo
        self.model = YOLO(MODEL_PATH)

        self.cap = None
        self.running = False
        self.thread = None

        self.fps = 0.0
        self.frame_count = 0
        self.start_time = time.perf_counter()

        # Estilo
        style = ttk.Style()
        style.theme_use("clam")
        style.configure(
            "TButton",
            font=("Arial", 11),
            padding=8
        )

        # Cabeçalho
        header = tk.Frame(root, bg="#171923")
        header.pack(fill="x", padx=20, pady=15)

        tk.Label(
            header,
            text="MONITORAMENTO INTELIGENTE",
            font=("Arial", 20, "bold"),
            fg="white",
            bg="#171923"
        ).pack(side="left")

        # Área de vídeo
        self.video_label = tk.Label(
            root,
            bg="#090b10",
            text="Clique em INICIAR para abrir a câmera",
            fg="white",
            font=("Arial", 14)
        )
        self.video_label.pack(
            fill="both",
            expand=True,
            padx=20,
            pady=10
        )

        # Painel de informações
        panel = tk.Frame(root, bg="#222634")
        panel.pack(fill="x", padx=20, pady=10)

        self.fps_label = tk.Label(
            panel,
            text="FPS: --",
            font=("Arial", 12, "bold"),
            fg="#4ade80",
            bg="#222634"
        )
        self.fps_label.pack(side="left", padx=20, pady=12)

        self.objects_label = tk.Label(
            panel,
            text="Objetos: 0",
            font=("Arial", 12, "bold"),
            fg="white",
            bg="#222634"
        )
        self.objects_label.pack(side="left", padx=20)

        self.status_label = tk.Label(
            panel,
            text="Câmera desligada",
            font=("Arial", 11),
            fg="#fbbf24",
            bg="#222634"
        )
        self.status_label.pack(side="right", padx=20)

        # Botões
        controls = tk.Frame(root, bg="#171923")
        controls.pack(pady=10)

        self.start_button = ttk.Button(
            controls,
            text="Iniciar câmera",
            command=self.start_camera
        )
        self.start_button.pack(side="left", padx=10)

        self.stop_button = ttk.Button(
            controls,
            text="Parar câmera",
            command=self.stop_camera,
            state="disabled"
        )
        self.stop_button.pack(side="left", padx=10)

        self.root.protocol("WM_DELETE_WINDOW", self.close)

    def start_camera(self):
        if self.running:
            return

        self.cap = cv2.VideoCapture(CAMERA_ID)

        if not self.cap.isOpened():
            self.cap.release()
            self.cap = None
            messagebox.showerror(
                "Erro",
                "Não foi possível abrir a câmera."
            )
            return

        self.running = True
        self.frame_count = 0
        self.start_time = time.perf_counter()

        self.start_button.config(state="disabled")
        self.stop_button.config(state="normal")
        self.status_label.config(text="Detectando objetos...")

        # A inferência roda em uma thread separada
        self.thread = threading.Thread(
            target=self.process_camera,
            daemon=True
        )
        self.thread.start()

    def process_camera(self):
        while self.running and self.cap is not None:
            success, frame = self.cap.read()

            if not success:
                break

            try:
                # Executa a detecção
                results = self.model.predict(
                    frame,
                    conf=CONFIDENCE,
                    verbose=False
                )

                # Desenha as bounding boxes
                annotated_frame = results[0].plot()

                # Conta as detecções
                object_count = len(results[0].boxes)

                # Calcula FPS médio aproximado
                self.frame_count += 1
                elapsed = time.perf_counter() - self.start_time
                fps = self.frame_count / max(elapsed, 1e-6)

                # Atualiza a interface pela thread principal
                self.root.after(
                    0,
                    self.update_interface,
                    annotated_frame,
                    object_count,
                    fps
                )

            except Exception as e:
                print("Erro na detecção:", e)
                break

        if self.cap is not None:
            self.cap.release()
            self.cap = None

        self.running = False
        self.root.after(0, self.camera_stopped)

    def update_interface(self, frame, object_count, fps):
        if not self.root.winfo_exists():
            return

        # OpenCV BGR -> RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Redimensiona mantendo a proporção
        image = Image.fromarray(frame_rgb)
        image.thumbnail((1000, 540))

        photo = ImageTk.PhotoImage(image=image)

        self.video_label.configure(image=photo, text="")
        self.video_label.image = photo

        self.fps_label.config(text=f"FPS: {fps:.1f}")
        self.objects_label.config(
            text=f"Objetos: {object_count}"
        )

    def camera_stopped(self):
        self.start_button.config(state="normal")
        self.stop_button.config(state="disabled")
        self.status_label.config(text="Câmera desligada")

    def stop_camera(self):
        self.running = False
        self.status_label.config(text="Parando câmera...")

    def close(self):
        self.running = False
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = YOLOInterface(root)
    root.mainloop()
