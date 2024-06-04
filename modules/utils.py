# modules/utils.py
import os
from datetime import datetime
import pytz

def create_save_folder(model_save_base_path, architecture):
    japan_timezone = pytz.timezone("Asia/Tokyo")
    now = datetime.now(japan_timezone)
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    folder_name = f"hyper_{timestamp}_{architecture}_{now.strftime('%H%M')}"
    save_path = os.path.join(model_save_base_path, folder_name)
    os.makedirs(save_path, exist_ok=True)
    return save_path
