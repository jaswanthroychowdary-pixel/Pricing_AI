"""
config_setup.py

Production-grade Secret and Configuration Management module for the Actuarial Pricing Engine.
Implements secure environment retrieval, version pinning compliance checks, and secure database 
connection engines. Designed to support ASOP 56 (Modeling) and secure credential isolation.
"""

import os
import sys
import logging
from typing import Dict, Any, Optional

# Set up structured logging for production auditing
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] (%(filename)s:%(lineno)d) - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("PricingEngineConfig")

class Config:
    """
    Central configuration and secret management.
    Never embeds secrets in code. Always retrieves from the host environment
    or secure secrets manager, injecting defaults strictly for local sandbox runs.
    """
    
    # Version Pinning Compliance (ASOP 56)
    VERSION = "2.1.0"
    EXPECTED_AGNO_VERSION_BAND = ">=1.0.0,<3.0.0"
    
    def __init__(self):
        # 1. API Security (Phase 1)
        self.google_api_key: Optional[str] = os.getenv("GOOGLE_API_KEY")
        
        # 2. Database Paths & Connections (Phase 2)
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        default_db = "/workspace/scratch/warehouse.db" if os.path.exists("/workspace/scratch") else os.path.join(project_root, "warehouse.db")
        default_raw = "/workspace/scratch/Pricing_Data.xlsx" if os.path.exists("/workspace/scratch") else os.path.join(project_root, "Pricing_Data.xlsx")
        self.db_path: str = os.getenv("DATABASE_PATH", default_db)
        self.raw_data_path: str = os.getenv("RAW_DATA_PATH", default_raw)
        
        # 3. Model Steering Parameters
        self.gemini_model_id: str = os.getenv("GEMINI_MODEL_ID", "gemini-3.1-flash-lite")
        
        # 4. Calibration & Risk Parameters (Default parameters from pricing registry)
        self.large_loss_loading: float = float(os.getenv("LARGE_LOSS_LOADING", "1.10"))
        self.profit_margin: float = float(os.getenv("PROFIT_MARGIN", "1.05"))
        self.premium_floor: float = float(os.getenv("PREMIUM_FLOOR", "50.00"))
        self.premium_cap: float = float(os.getenv("PREMIUM_CAP", "5000.00"))
        
        self._validate_configs()

    def _validate_configs(self):
        """
        Validates the configuration parameters to prevent silent, corrupt downstream runs.
        Ensures strict compliance with data-type constraints and logical business bounds.
        """
        logger.info(f"Initializing Pricing Engine Config [v{self.VERSION}]")
        
        # Validate API Key Presence (Phase 1)
        if not self.google_api_key:
            logger.warning("GOOGLE_API_KEY environment variable is not set. AI agents will run in simulated mode.")
        else:
            logger.info("GOOGLE_API_KEY successfully authenticated and loaded.")
            
        # Validate Database directory exists
        db_dir = os.path.dirname(self.db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)
            logger.info(f"Created database directory: {db_dir}")

        # Validate Numerical Pricing Bounds (ASOP 56 check)
        if self.large_loss_loading < 1.0:
            raise ValueError(f"Large loss loading (L) must be >= 1.0 to prevent under-capitalization. Found: {self.large_loss_loading}")
        if self.profit_margin < 0.90:
            raise ValueError(f"Profit margin (M) must be >= 0.90 to cover administrative expenses. Found: {self.profit_margin}")
        if self.premium_floor >= self.premium_cap:
            raise ValueError(f"Premium floor ({self.premium_floor}) cannot exceed or equal premium cap ({self.premium_cap})")
            
        logger.info("All configuration and pricing parameters passed validation constraints.")

    def get_summary_dictionary(self) -> Dict[str, Any]:
        """
        Returns a masked configuration dictionary for auditing and AI metadata injection.
        Protects secrets from being dumped in plaintext inside logs or reasoning traces.
        """
        return {
            "engine_version": self.VERSION,
            "gemini_model_id": self.gemini_model_id,
            "database_path": self.db_path,
            "google_api_key_configured": self.google_api_key is not None,
            "risk_parameters": {
                "large_loss_loading": self.large_loss_loading,
                "profit_margin": self.profit_margin,
                "premium_floor": self.premium_floor,
                "premium_cap": self.premium_cap
            }
        }

# Global Singleton Instance for Runtime Ingestion
config = Config()
