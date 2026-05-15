import logging
import sys
import os

def setup_logging():
    """Configura el logging básico para la aplicación, guardando en archivo y consola."""
    # Desactivar handlers previos si los hay
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
        
    # Crear carpeta logs si no existe en la raíz del proyecto
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    logs_dir = os.path.join(base_dir, 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    
    log_file = os.path.join(logs_dir, 'pipeline.log')
        
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),          
            logging.FileHandler(log_file, mode='a')    
        ]
    )
    return logging.getLogger("TrainAnomalyDetection")
