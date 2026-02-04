"""
Configuration Manager for Railway Crossing System
Manages crossings, cameras, and PLC configurations
"""

import json
import yaml
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime


class ConfigManager:
    """Manages system configuration for crossings, cameras, and PLCs"""

    def __init__(self, config_file: str = "gui_config.json"):
        self.config_file = Path(config_file)
        self.config = self._load_config()

    def _load_config(self) -> Dict:
        """Load configuration from file"""
        if self.config_file.exists():
            with open(self.config_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            # Default configuration
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
        """Save configuration to file"""
        self.config["last_updated"] = datetime.now().isoformat()
        with open(self.config_file, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)

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
                camera_data["id"] = camera_id
                camera_data["updated_at"] = datetime.now().isoformat()
                crossing["cameras"][i] = camera_data
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
