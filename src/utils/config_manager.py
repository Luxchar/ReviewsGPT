# ============================================================================
# GESTIONNAIRE DE CONFIGURATION YAML
# ============================================================================

import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional
import logging
from datetime import datetime

class ConfigManager:
    """
    Gestionnaire de configuration pour le scraping robuste
    Charge et valide les configurations depuis un fichier YAML
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or self._get_default_config_path()
        self.config = {}
        self.logger = self._setup_logging()
        
    def _get_default_config_path(self) -> str:
        """Retourne le chemin par défaut du fichier de configuration"""
        current_dir = Path(__file__).parent
        return str(current_dir.parent / "configs" / "scraping_config.yaml")
    
    def _setup_logging(self) -> logging.Logger:
        """Configure le logger"""
        logger = logging.getLogger("ConfigManager")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '[%(asctime)s] %(name)s - %(levelname)s - %(message)s',
                datefmt='%H:%M:%S'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def load_config(self) -> Dict[str, Any]:
        """
        Charge la configuration depuis le fichier YAML
        """
        try:
            if not os.path.exists(self.config_path):
                self.logger.error(f"Fichier de configuration non trouvé: {self.config_path}")
                return self._get_default_config()
            
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
            
            # Validation de base
            self._validate_config()
            
            self.logger.info(f"Configuration chargée: {self.config_path}")
            return self.config
            
        except yaml.YAMLError as e:
            self.logger.error(f"Erreur de parsing YAML: {e}")
            return self._get_default_config()
        except Exception as e:
            self.logger.error(f"Erreur chargement configuration: {e}")
            return self._get_default_config()
    
    def _validate_config(self):
        """Valide la configuration chargée"""
        required_sections = ['scraping', 'driver', 'scout', 'scraper', 'output', 'logging']
        
        for section in required_sections:
            if section not in self.config:
                self.logger.warning(f"Section manquante dans la configuration: {section}")
                self.config[section] = {}
        
        # Validation des valeurs critiques
        scraping = self.config.get('scraping', {})
        
        if scraping.get('site') not in ['amazon', 'ebay']:
            self.logger.warning(f"Site non supporté: {scraping.get('site')}, utilisation d'Amazon par défaut")
            scraping['site'] = 'amazon'
        
        if not isinstance(scraping.get('max_products'), int) or scraping.get('max_products') <= 0:
            self.logger.warning("max_products invalide, utilisation de 5 par défaut")
            scraping['max_products'] = 5
        
        if not isinstance(scraping.get('reviews_per_rating'), int) or scraping.get('reviews_per_rating') <= 0:
            self.logger.warning("reviews_per_rating invalide, utilisation de 20 par défaut")
            scraping['reviews_per_rating'] = 20
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Retourne une configuration par défaut en cas d'erreur"""
        self.logger.info("Utilisation de la configuration par défaut")
        
        return {
            'scraping': {
                'site': 'amazon',
                'category': 'laptop',
                'max_products': 5,
                'reviews_per_rating': 20,
                'headless': True,
                'timeout': 60
            },
            'driver': {
                'max_attempts': 3,
                'retry_delay': 3,
                'user_agent': '',
                'proxy': ''
            },
            'scout': {
                'detection_timeout': 30,
                'max_pages_scout': 3,
                'auto_save_selectors': True
            },
            'scraper': {
                'max_pages_per_product': 5,
                'delay_between_pages': 3,
                'delay_between_pages_max': 6,
                'delay_between_products': 5,
                'min_review_length': 15,
                'remove_duplicates': True
            },
            'output': {
                'data_dir': '../data/raw',
                'logs_dir': '../logs',
                'config_dir': '../config',
                'timestamp_format': '%Y%m%d_%H%M%S',
                'save_csv': True,
                'save_json': True,
                'file_prefix': 'reviews'
            },
            'logging': {
                'level': 'INFO',
                'show_timestamp': True,
                'save_to_file': True,
                'show_traceback': True
            },
            'advanced': {
                'use_ultra_secure_scout': True,
                'rotate_user_agents': True,
                'use_stealth_scripts': True,
                'save_error_screenshots': False,
                'memory_limit_mb': 500
            }
        }
    
    def get_example_config(self, example_name: str) -> Optional[Dict[str, Any]]:
        """
        Retourne la configuration d'un exemple spécifique
        """
        examples = self.config.get('examples', {})
        
        if example_name in examples:
            # Fusionner avec la configuration de base
            base_config = self.config.copy()
            example_config = examples[example_name]
            
            # Mettre à jour la section scraping avec les valeurs de l'exemple
            base_config['scraping'].update(example_config)
            
            self.logger.info(f"Configuration exemple '{example_name}' chargée")
            return base_config
        else:
            self.logger.error(f"Exemple '{example_name}' non trouvé")
            available_examples = list(examples.keys())
            self.logger.info(f"Exemples disponibles: {available_examples}")
            return None
    
    def create_sample_config(self, output_path: Optional[str] = None):
        """
        Crée un fichier de configuration d'exemple
        """
        if output_path is None:
            output_path = "scraping_config_sample.yaml"
        
        sample_config = self._get_default_config()
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                yaml.dump(sample_config, f, default_flow_style=False, indent=2, allow_unicode=True)
            
            self.logger.info(f"Fichier de configuration d'exemple créé: {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Erreur création fichier d'exemple: {e}")
            return False
    
    def get_scraping_params(self) -> Dict[str, Any]:
        """
        Retourne les paramètres formatés pour le workflow de scraping
        """
        if not self.config:
            self.load_config()
        
        scraping = self.config.get('scraping', {})
        
        return {
            'category': scraping.get('category', 'laptop'),
            'site': scraping.get('site', 'amazon'),
            'max_products': scraping.get('max_products', 5),
            'reviews_per_rating': scraping.get('reviews_per_rating', 20),
            'headless': scraping.get('headless', True),
            'timeout': scraping.get('timeout', 60)
        }
    
    def get_driver_params(self) -> Dict[str, Any]:
        """
        Retourne les paramètres du driver
        """
        if not self.config:
            self.load_config()
        
        return self.config.get('driver', {})
    
    def get_scout_params(self) -> Dict[str, Any]:
        """
        Retourne les paramètres du scout
        """
        if not self.config:
            self.load_config()
        
        return self.config.get('scout', {})
    
    def get_scraper_params(self) -> Dict[str, Any]:
        """
        Retourne les paramètres du scraper
        """
        if not self.config:
            self.load_config()
        
        return self.config.get('scraper', {})
    
    def get_output_params(self) -> Dict[str, Any]:
        """
        Retourne les paramètres de sortie
        """
        if not self.config:
            self.load_config()
        
        return self.config.get('output', {})
    
    def get_logging_params(self) -> Dict[str, Any]:
        """
        Retourne les paramètres de logging
        """
        if not self.config:
            self.load_config()
        
        return self.config.get('logging', {})
    
    def list_examples(self) -> list:
        """
        Liste tous les exemples disponibles
        """
        if not self.config:
            self.load_config()
        
        examples = self.config.get('examples', {})
        return list(examples.keys())
    
    def print_config_summary(self):
        """
        Affiche un résumé de la configuration actuelle
        """
        if not self.config:
            self.load_config()
        
        scraping = self.config.get('scraping', {})
        
        print("📋 RÉSUMÉ DE LA CONFIGURATION")
        print("=" * 50)
        print(f"🌐 Site: {scraping.get('site', 'N/A')}")
        print(f"📦 Catégorie: {scraping.get('category', 'N/A')}")
        print(f"📊 Max produits: {scraping.get('max_products', 'N/A')}")
        print(f"⭐ Reviews par note: {scraping.get('reviews_per_rating', 'N/A')}")
        print(f"👁️ Mode headless: {scraping.get('headless', 'N/A')}")
        print(f"⏱️ Timeout: {scraping.get('timeout', 'N/A')}s")
        
        examples = self.list_examples()
        if examples:
            print(f"\n🧪 Exemples disponibles: {', '.join(examples)}")

if __name__ == "__main__":
    # Test du gestionnaire de configuration
    config_manager = ConfigManager()
    config = config_manager.load_config()
    config_manager.print_config_summary()
