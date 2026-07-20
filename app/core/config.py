"""
Configuration Manager for Railway Crossing System
Manages crossings, cameras, and PLC configurations
"""

import copy
import json
import yaml
import os
import tempfile
import logging
import threading
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

logger = logging.getLogger("RailSafe.config")

# app/core/config.py → app/core → app → project_root
_PROJECT_ROOT  = Path(__file__).parent.parent.parent
_CONFIG_DIR    = _PROJECT_ROOT / "config"
_GUI_CONFIG    = _CONFIG_DIR / "gui_config.json"
_APP_CONFIG    = _CONFIG_DIR / "config.yaml"
_CAMERA_STATE  = _CONFIG_DIR / "camera_state.json"


class ConfigManager:
    """Manages system configuration for crossings, cameras, and PLCs"""

    def __init__(self, config_file: str = None):
        # Bir nechta threaddan yoziladi/o'qiladi — RLock bilan himoyalanadi.
        # RLock reentrant: bir thread ichida ichma-ich chaqirilishi mumkin (deadlock yo'q).
        self._lock = threading.RLock()
        self.config_file = Path(config_file) if config_file else _GUI_CONFIG
        self.config = self._load_config()
        self._camera_state_cache = None  # Disk o'qishni kamaytirish uchun xotira keshi

    def _load_config(self) -> Dict:
        """Load configuration from file"""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error("Config o'qishda xato: %s, default ishlatiladi", e)
        return {
            "crossings": [],
            "settings": {
                "theme": "dark",
                "language": "uz",
                "warning_threshold": 10.0,
                "violation_threshold": 15.0
            },
            "last_updated": datetime.now().isoformat()
        }

    def save_config(self):
        """Atomic write: avval temp faylga yoziladi, keyin rename — buzilish xavfsiz."""
        with self._lock:
            self.config["last_updated"] = datetime.now().isoformat()
            tmp_path = None
            try:
                self.config_file.parent.mkdir(parents=True, exist_ok=True)
                fd, tmp_path = tempfile.mkstemp(
                    dir=self.config_file.parent, suffix=".tmp", prefix=".cfg_")
                with os.fdopen(fd, 'w', encoding='utf-8') as f:
                    json.dump(self.config, f, indent=2, ensure_ascii=False)
                os.replace(tmp_path, self.config_file)
            except Exception as e:
                logger.error("Saqlashda xato: %s", e)
                if tmp_path:
                    try:
                        os.unlink(tmp_path)
                    except Exception:
                        pass

    def _next_id(self, counter_key: str, existing_ids: List[int]) -> int:
        """Monotonik ID beruvchi. config ichida `counter_key` (masalan
        next_crossing_id) saqlanadi — o'chirilgan-keyin-qayta qo'shilgan yozuv
        hech qachon eski ID ni qayta ishlatmaydi, shu bilan tarixiy statistika
        boshqa yozuvga o'tib ketmaydi.
        """
        # Migratsiya: eski configda counter bo'lmasa, mavjud maksimal + 1 dan boshlaymiz
        cur_max = max(existing_ids) if existing_ids else 0
        stored = self.config.get(counter_key, 0)
        new_id = max(stored, cur_max + 1)
        self.config[counter_key] = new_id + 1
        return new_id

    def get_crossings(self) -> List[Dict]:
        """Get all crossings.

        Chuqur nusxa qaytaradi — chaqiruvchi (API/push fon oqimlari) iteratsiya
        qilayotganda GUI oqimi ro'yxatni o'zgartirsa "list changed size during
        iteration" bo'lmasligi uchun. Ichki mutatsiya qiluvchi metodlar bevosita
        self.config ustida ishlaydi, shu sababli bu ularga ta'sir qilmaydi.
        """
        with self._lock:
            return copy.deepcopy(self.config.get("crossings", []))

    def get_crossing(self, crossing_id: int) -> Optional[Dict]:
        """Get a specific crossing by ID (chuqur nusxa — get_crossings bilan bir xil sabab)."""
        with self._lock:
            for crossing in self.config.get("crossings", []):
                if crossing.get("id") == crossing_id:
                    return copy.deepcopy(crossing)
        return None

    def add_crossing(self, crossing_data: Dict) -> int:
        """Add a new crossing"""
        with self._lock:
            # Monotonik ID — o'chirilgan ID lar qayta ishlatilmaydi
            existing_ids = [c.get("id") for c in self.config.get("crossings", [])
                            if c.get("id") is not None]
            new_id = self._next_id("next_crossing_id", existing_ids)

            crossing_data["id"] = new_id
            crossing_data["created_at"] = datetime.now().isoformat()
            crossing_data["status"] = "offline"

            # Ensure cameras and plc exist
            if "cameras" not in crossing_data:
                crossing_data["cameras"] = []
            if "plc" not in crossing_data:
                crossing_data["plc"] = {
                    "ip": "",
                    "port": 102,
                    "enabled": False
                }

            self.config.setdefault("crossings", []).append(crossing_data)
            self.save_config()
            return new_id

    def update_crossing(self, crossing_id: int, crossing_data: Dict) -> bool:
        """Update an existing crossing"""
        with self._lock:
            for i, crossing in enumerate(self.config.get("crossings", [])):
                if crossing.get("id") == crossing_id:
                    crossing_data["id"] = crossing_id
                    crossing_data["updated_at"] = datetime.now().isoformat()
                    self.config["crossings"][i] = crossing_data
                    self.save_config()
                    return True
        return False

    def delete_crossing(self, crossing_id: int) -> bool:
        """Delete a crossing"""
        with self._lock:
            crossings = self.config.get("crossings", [])
            for i, crossing in enumerate(crossings):
                if crossing.get("id") == crossing_id:
                    del crossings[i]
                    self.save_config()
                    return True
        return False

    def add_camera(self, crossing_id: int, camera_data: Dict) -> Optional[int]:
        """Add a camera to a crossing"""
        with self._lock:
            crossing = self.get_crossing(crossing_id)
            if not crossing:
                return None

            # Monotonik kamera ID — pereezd ichida saqlanadi (next_camera_id)
            existing_ids = [c.get("id") for c in crossing.get("cameras", [])
                            if c.get("id") is not None]
            cur_max = max(existing_ids) if existing_ids else 0
            stored = crossing.get("next_camera_id", 0)
            new_id = max(stored, cur_max + 1)
            crossing["next_camera_id"] = new_id + 1

            camera_data["id"] = new_id
            camera_data["created_at"] = datetime.now().isoformat()
            camera_data["status"] = "offline"

            crossing.setdefault("cameras", []).append(camera_data)
            self.update_crossing(crossing_id, crossing)
            return new_id

    def update_camera(self, crossing_id: int, camera_id: int, camera_data: Dict) -> bool:
        """Update a camera in a crossing"""
        with self._lock:
            crossing = self.get_crossing(crossing_id)
            if not crossing:
                return False

            for i, camera in enumerate(crossing.get("cameras", [])):
                if camera.get("id") == camera_id:
                    crossing["cameras"][i].update(camera_data)
                    crossing["cameras"][i]["id"] = camera_id
                    crossing["cameras"][i]["updated_at"] = datetime.now().isoformat()
                    self.update_crossing(crossing_id, crossing)
                    return True
        return False

    def delete_camera(self, crossing_id: int, camera_id: int) -> bool:
        """Delete a camera from a crossing"""
        with self._lock:
            crossing = self.get_crossing(crossing_id)
            if not crossing:
                return False

            cameras = crossing.get("cameras", [])
            for i, camera in enumerate(cameras):
                if camera.get("id") == camera_id:
                    del cameras[i]
                    self.update_crossing(crossing_id, crossing)
                    return True
        return False

    def update_plc(self, crossing_id: int, plc_data: Dict) -> bool:
        """Update PLC configuration for a crossing"""
        with self._lock:
            crossing = self.get_crossing(crossing_id)
            if not crossing:
                return False

            crossing["plc"] = plc_data
            self.update_crossing(crossing_id, crossing)
            return True

    def get_settings(self) -> Dict:
        """Get application settings (chuqur nusxa — fon oqimlari xavfsiz o'qishi uchun)."""
        with self._lock:
            return copy.deepcopy(self.config.get("settings", {}))

    def update_settings(self, settings: Dict):
        """Update application settings"""
        with self._lock:
            self.config.setdefault("settings", {}).update(settings)
            self.save_config()

    # --- Kamera holati (paused/resumed) — config/camera_state.json ---

    def _load_camera_state(self) -> Dict:
        with self._lock:
            if self._camera_state_cache is not None:
                return self._camera_state_cache
            try:
                if _CAMERA_STATE.exists():
                    with open(_CAMERA_STATE, 'r', encoding='utf-8') as f:
                        self._camera_state_cache = json.load(f)
                        return self._camera_state_cache
            except Exception:
                pass
            self._camera_state_cache = {"paused": {}}
            return self._camera_state_cache

    def _save_camera_state(self, state: Dict):
        with self._lock:
            self._camera_state_cache = state
            _CONFIG_DIR.mkdir(parents=True, exist_ok=True)
            tmp_path = None
            try:
                fd, tmp_path = tempfile.mkstemp(dir=_CONFIG_DIR, suffix=".tmp")
                with os.fdopen(fd, 'w', encoding='utf-8') as f:
                    json.dump(state, f, indent=2)
                os.replace(tmp_path, _CAMERA_STATE)
            except Exception as e:
                logger.error("Camera state saqlashda xato: %s", e)
                if tmp_path:
                    try:
                        os.unlink(tmp_path)
                    except Exception:
                        pass

    def get_paused_cameras(self, crossing_id: int) -> set:
        """Berilgan pereezd uchun to'xtatilgan kamera ID larini qaytaradi."""
        with self._lock:
            state = self._load_camera_state()
            paused = state.get("paused", {}).get(str(crossing_id), [])
            return set(paused)

    def set_camera_paused(self, crossing_id: int, camera_id: int, paused: bool):
        """Kamerani paused/resumed holatini saqlaydi."""
        with self._lock:
            state = self._load_camera_state()
            key = str(crossing_id)
            paused_list = state.setdefault("paused", {}).get(key, [])
            if paused:
                if camera_id not in paused_list:
                    paused_list.append(camera_id)
            else:
                paused_list = [c for c in paused_list if c != camera_id]
            state["paused"][key] = paused_list
            self._save_camera_state(state)

    def export_to_yaml(self, crossing_id: int, output_file: str) -> bool:
        """Export crossing configuration to YAML (for backend processing)"""
        crossing = self.get_crossing(crossing_id)
        if not crossing:
            return False

        # Haqiqiy sozlangan model yo'lini olamiz (placeholder emas)
        model_path = self.get_car_detector_config().get(
            "model_path", str(_PROJECT_ROOT / "models" / "yolo26m.pt"))

        settings = self.get_settings()

        # Convert to backend format (config.yaml structure)
        backend_config = {
            "model": {
                "path": model_path,
                "target_classes": [2, 5, 7],
                "class_names": {
                    2: "Yengil avtomobil",
                    3: "Mototsikl",
                    5: "Avtobus",
                    6: "Poyezd",
                    7: "Yuk mashinasi"
                }
            },
            "plc": crossing.get("plc", {}),
            "thresholds": {
                "warning": settings.get("warning_threshold", 10.0),
                "violation": settings.get("violation_threshold", 15.0)
            },
            "processing": {
                "adaptive_mode": True,
                "frame_skip_idle": 3,
                "frame_skip_active": 1,
                "polygon_length": 8.0
            },
            "cameras": [
                {
                    "id": cam.get("id"),
                    "name": cam.get("name", ""),
                    "source": cam.get("source", ""),
                    "polygon_file": cam.get("polygon_file", ""),
                    "enabled": cam.get("enabled", False)
                }
                for cam in crossing.get("cameras", [])
            ]
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            yaml.dump(backend_config, f, allow_unicode=True, default_flow_style=False)

        return True

    def import_from_yaml(self, yaml_file: str) -> Optional[int]:
        """Import crossing configuration from YAML"""
        try:
            with open(yaml_file, 'r', encoding='utf-8') as f:
                backend_config = yaml.safe_load(f)

            # Convert from backend format
            crossing_data = {
                "name": f"Imported Crossing - {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                "location": "Unknown",
                "cameras": [
                    {
                        "id": cam.get("id"),
                        "name": cam.get("name", ""),
                        "source": cam.get("source", ""),
                        "polygon_file": cam.get("polygon_file", ""),
                        "enabled": cam.get("enabled", False),
                        "type": "main" if cam.get("id") == 1 else "additional"
                    }
                    for cam in backend_config.get("cameras", [])
                ],
                "plc": backend_config.get("plc", {})
            }

            return self.add_crossing(crossing_data)
        except Exception as e:
            logger.error("Error importing YAML: %s", e)
            return None

    def get_car_detector_config(self, config_yaml_path: str = None) -> Dict:
        """
        Get car detector configuration from config/config.yaml.
        GUI settings dan model_type va custom_model_path tekshiriladi.
        """
        default_config = {
            "enabled": False,
            "model_path": str(_PROJECT_ROOT / "models" / "yolo26m.pt"),
            "confidence": 0.3,
            "iou_threshold": 0.45,
            "imgsz": 640,
            "device": "cuda",
            "half": True,
            "stream": True
        }

        try:
            config_path = Path(config_yaml_path) if config_yaml_path else _APP_CONFIG
            if not config_path.is_absolute():
                config_path = _PROJECT_ROOT / config_path

            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    yaml_config = yaml.safe_load(f)

                if yaml_config and "car_detector" in yaml_config:
                    car_config = yaml_config["car_detector"]
                    # Make model_path absolute if relative
                    if "model_path" in car_config:
                        model_path = Path(car_config["model_path"])
                        if not model_path.is_absolute():
                            car_config["model_path"] = str(_PROJECT_ROOT / model_path)

                    # GUI settings dan maxsus model tekshirish
                    gui_settings = self.get_settings()
                    if gui_settings.get("model_type") == "custom":
                        # config.yaml dan custom_model_path o'qish
                        raw_custom = car_config.get("custom_model_path", "")
                        custom_path = Path(raw_custom)
                        if not custom_path.is_absolute():
                            custom_path = _PROJECT_ROOT / custom_path
                        if custom_path.is_file():
                            car_config["model_path"] = str(custom_path)
                            car_config["imgsz"] = car_config.get("custom_imgsz", 1088)
                            car_config["filter_classes"] = car_config.get("custom_filter_classes")
                            car_config["is_custom_model"] = True
                            logger.info("Maxsus model: %s", custom_path)
                    else:
                        car_config["is_custom_model"] = False

                    return car_config

        except Exception as e:
            logger.error("Error loading car detector config: %s", e)

        return default_config
