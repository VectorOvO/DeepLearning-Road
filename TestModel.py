# TestModel_preprocess.py
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import os
from PIL import Image, ImageChops, ImageOps
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import ToTensor, Normalize, Compose
from DeepLearning import CNN   # 把 this 改成定义 CNN 的文件名，比如 DeepLearning (no .py)

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load model
model = CNN(1)   # 和训练时一致
model.load_state_dict(torch.load("mnist_cnn.pth", map_location=device))
model.to(device)
model.eval()

# transforms (final)
final_transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])

def preprocess_image(filepath):
    # 打开灰度图
    img = Image.open(filepath).convert("L")

    # 反色（黑底白字 → 白底黑字，MNIST 格式要求）
    img = ImageOps.invert(img)

    # 转 numpy，找到非空区域（去掉多余的白边）
    arr = np.array(img)
    coords = np.argwhere(arr > 0)  # 非白色区域
    if coords.size == 0:
        raise ValueError("输入图片是空白的！")
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)
    crop = arr[y0:y1+1, x0:x1+1]

    # 保持纵横比缩放到 20x20
    h, w = crop.shape
    if h > w:
        new_h = 20
        new_w = int(round((w * 20.0) / h))
    else:
        new_w = 20
        new_h = int(round((h * 20.0) / w))

    # ✅ Pillow ≥10 推荐写法 (Resampling.LANCZOS)
    crop_img = Image.fromarray(crop).resize((new_w, new_h), Image.Resampling.LANCZOS)

    # 填充到 28x28
    new_img = Image.new("L", (28, 28), 0)
    left = (28 - new_w) // 2
    top = (28 - new_h) // 2
    new_img.paste(crop_img, (left, top))

    return new_img

def predict_image(model, filepath, device):
    pil_img = preprocess_image(filepath)          # PIL 28x28 grayscale
    tensor = final_transform(pil_img).unsqueeze(0).to(device)  # [1,1,28,28]
    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
        pred = int(probs.argmax())
    # show top-3 probs
    top3_idx = probs.argsort()[::-1][:3]
    top3 = [(int(i), float(probs[i])) for i in top3_idx]
    # display
    #print(f"Prediction: {pred}, top-3: {top3}")
    print(f"你写的字母应该是：{pred}")
    pil_img.show(title=f"Pred:{pred}")   # 打开图片窗口查看处理后图像
    return pred, top3

# ==== 用法示例 ====
if __name__ == "__main__":
    img_path = "digit3.png"   # 或者 digit.png，确保文件存在
    if not os.path.exists(img_path):
        raise SystemExit(f"{img_path} not found. Put your drawn image in the script folder.")
    pred, top3 = predict_image(model, img_path, device)
    # 此时如果跑完了在Terminal里输入
    # conda activate DeepLearningPro
    # tensorboard --logdir=runs
    # 点击 http://localhost:6006/ 即可查看损失曲线及其精确度曲线
