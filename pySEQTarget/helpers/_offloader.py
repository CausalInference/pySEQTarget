from pathlib import Path
from typing import Any, Optional, Union
import joblib
import polars as pl


class Offloader:
    """Manages disk-based storage for models and intermediate data"""
    
    def __init__(self, 
                 enabled: bool, 
                 dir: str, 
                 compression: int = 3
                 ):
        self.enabled = enabled
        self.dir = Path(dir)
        self.compression = compression
    
    def save_model(self, model: Any, name: str, boot_idx: Optional[int] = None) -> Union[Any, str]:
        """Save a fitted model to disk and return a reference"""
        if not self.enabled:
            return model
        
        filename = f"{name}_boot{boot_idx}.pkl" if boot_idx is not None else f"{name}.pkl"
        filepath = self.dir / filename
        
        joblib.dump(model, filepath, compress=self.compression)
        
        return str(filepath)
    
    def load_model(self, ref: Union[Any, str]) -> Any:
        if not self.enabled or not isinstance(ref, str):
            return ref
        
        return joblib.load(ref)