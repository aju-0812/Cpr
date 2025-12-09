import yaml
from dataclasses import dataclass

@dataclass
class Marker:
    id: int
    x: float
    y: float
    type: str

class MarkerMap:
    def __init__(self, config_source):
        if isinstance(config_source, str):
            with open(config_source, 'r') as f:
                cfg = yaml.safe_load(f)
        elif isinstance(config_source, dict):
            cfg = config_source
        else:
            raise ValueError("config_source must be a file path or a dictionary")
            
        self.markers = [Marker(**m) for m in cfg.get('markers', [])]

    def get_marker(self, id):
        for m in self.markers:
            if m.id == id:
                return m
        return None
