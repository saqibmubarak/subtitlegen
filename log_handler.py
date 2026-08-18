import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('subtitle_generator.log')
    ]
)
logger = logging.getLogger(__name__)

SRT_FILE_EXTENSIONS = (".srt", ".vtt")

