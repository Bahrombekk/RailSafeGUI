"""
Configuration Manager for Railway Crossing System
Manages crossings, cameras, and PLC configurations
"""

import json
import yaml
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

# gui/utils/config_manager.py → gui/utils → gui → project_root
_PROJECT_ROOT  = Path(__file__).parent.parent.parent
_CONFIG_DIR    = _PROJECT_ROOT / "config"
_GUI_CONFIG    = _CONFIG_DIR / "gui_config.json"
_APP_CONFIG    = _CONFIG_DIR / "config.yaml"
_CAMERA_STATE  = _CONFIG_DIR / "camera_state.json"


class ConfigManager:
    """Manages system configuration for crossings, cameras, and PLCs"""

    def __init__(self, config_file: str = None):
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
                print(f"[ConfigManager] Config o'qishda xato: {e}, default ishlatiladi")
        return {
            "crossings": [],
            "settings": {
                "theme": "dark",
                "language": "uz",
                "auto_save": True,
                "warning_threshold": 10.0,
                "violation_threshold": 15.0
            },
            "last_updated": datetime.now().isoformat()
        }

    def save_config(self):
        """Atomic write: avval temp faylga yoziladi, keyin rename — buzilish xavfsiz."""
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
            print(f"[ConfigManager] Saqlashda xato: {e}")
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

    def get_crossings(self) -> List[Dict]:
        """Get all crossings"""
        return self.config.get("crossings", [])

    def get_crossing(self, crossing_id: int) -> Optional[Dict]:
        """Get a specific crossing by ID"""
        for crossing in self.config.get("crossings", []):
            if crossing["id"] == crossing_id:
                return crossing
        return None

    def add_crossing(self, crossing_data: Dict) -> int:
        """Add a new crossing"""
        # Generate new ID
        existing_ids = [c["id"] for c in self.config.get("crossings", [])]
        new_id = max(existing_ids) + 1 if existing_ids else 1

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

        self.config["crossings"].append(crossing_data)
        self.save_config()
        return new_id

    def update_crossing(self, crossing_id: int, crossing_data: Dict) -> bool:
        """Update an existing crossing"""
        for i, crossing in enumerate(self.config.get("crossings", [])):
            if crossing["id"] == crossing_id:
                crossing_data["id"] = crossing_id
                crossing_data["updated_at"] = datetime.now().isoformat()
                self.config["crossings"][i] = crossing_data
                self.save_config()
                return True
        return False

    def delete_crossing(self, crossing_id: int) -> bool:
        """Delete a crossing"""
        crossings = self.config.get("crossings", [])
        for i, crossing in enumerate(crossings):
            if crossing["id"] == crossing_id:
                del crossings[i]
                self.save_config()
                return True
        return False

    def add_camera(self, crossing_id: int, camera_data: Dict) -> Optional[int]:
        """Add a camera to a crossing"""
        crossing = self.get_crossing(crossing_id)
        if not crossing:
            return None

        # Generate camera ID
        existing_ids = [c["id"] for c in crossing.get("cameras", [])]
        new_id = max(existing_ids) + 1 if existing_ids else 1

        camera_data["id"] = new_id
        camera_data["created_at"] = datetime.now().isoformat()
        camera_data["status"] = "offline"

        crossing["cameras"].append(camera_data)
        self.update_crossing(crossing_id, crossing)
        return new_id

    def update_camera(self, crossing_id: int, camera_id: int, camera_data: Dict) -> bool:
        """Update a camera in a crossing"""
        crossing = self.get_crossing(crossing_id)
        if not crossing:
            return False

        for i, camera in enumerate(crossing.get("cameras", [])):
            if camera["id"] == camera_id:
                crossing["cameras"][i].update(camera_data)
                crossing["cameras"][i]["id"] = camera_id
                crossing["cameras"][i]["updated_at"] = datetime.now().isoformat()
                self.update_crossing(crossing_id, crossing)
                return True
        return False

    def delete_camera(self, crossing_id: int, camera_id: int) -> bool:
        """Delete a camera from a crossing"""
        crossing = self.get_crossing(crossing_id)
        if not crossing:
            return False

        cameras = crossing.get("cameras", [])
        for i, camera in enumerate(cameras):
            if camera["id"] == camera_id:
                del cameras[i]
                self.update_crossing(crossing_id, crossing)
                return True
        return False

    def update_plc(self, crossing_id: int, plc_data: Dict) -> bool:
        """Update PLC configuration for a crossing"""
        crossing = self.get_crossing(crossing_id)
        if not crossing:
            return False

        crossing["plc"] = plc_data
        self.update_crossing(crossing_id, crossing)
        return True

    def get_settings(self) -> Dict:
        """Get application settings"""
        return self.config.get("settings", {})

    def update_settings(self, settings: Dict):
        """Update application settings"""
        self.config["settings"].update(settings)
        self.save_config()

    # --- Kamera holati (paused/resumed) — config/camera_state.json ---

    def _load_camera_state(self) -> Dict:
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
        self._camera_state_cache = state
        _CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        tmp_path = None
        try:
            fd, tmp_path = tempfile.mkstemp(dir=_CONFIG_DIR, suffix=".tmp")
            with os.fdopen(fd, 'w', encoding='utf-8') as f:
                json.dump(state, f, indent=2)
            os.replace(tmp_path, _CAMERA_STATE)
        except Exception as e:
            print(f"[ConfigManager] Camera state saqlashda xato: {e}")
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

    def get_paused_cameras(self, crossing_id: int) -> set:
        """Berilgan pereezd uchun to'xtatilgan kamera ID larini qaytaradi."""
        state = self._load_camera_state()
        paused = state.get("paused", {}).get(str(crossing_id), [])
        return set(paused)

    def set_camera_paused(self, crossing_id: int, camera_id: int, paused: bool):
        """Kamerani paused/resumed holatini saqlaydi."""
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

        # Convert to backend format (config.yaml structure)
        backend_config = {
            "model": {
                "path": "/path/to/yolo/model.pt",
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
                "warning": self.config["settings"].get("warning_threshold", 10.0),
                "violation": self.config["settings"].get("violation_threshold", 15.0)
            },
            "processing": {
                "adaptive_mode": True,
                "frame_skip_idle": 3,
                "frame_skip_active": 1,
                "polygon_length": 8.0
            },
            "cameras": [
                {
                    "id": cam["id"],
                    "name": cam["name"],
                    "source": cam["source"],
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
                        "id": cam["id"],
                        "name": cam["name"],
                        "source": cam["source"],
                        "polygon_file": cam.get("polygon_file", ""),
                        "enabled": cam.get("enabled", False),
                        "type": "main" if cam["id"] == 1 else "additional"
                    }
                    for cam in backend_config.get("cameras", [])
                ],
                "plc": backend_config.get("plc", {})
            }

            return self.add_crossing(crossing_data)
        except Exception as e:
            print(f"Error importing YAML: {e}")
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
                            print(f"[ConfigManager] Maxsus model: {custom_path}")
                    else:
                        car_config["is_custom_model"] = False

                    return car_config

        except Exception as e:
            print(f"[ConfigManager] Error loading car detector config: {e}")

        return default_config
