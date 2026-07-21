import os
import logging

def setup_logging(verbosity: str):
    log_levels = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }
    log_level = log_levels.get(verbosity.upper(), logging.INFO)
    # Ensure logs directory exists
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "eqbench3.log")

    # Remove existing handlers to avoid duplicate logs if called multiple times
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - [%(threadName)s] - %(filename)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler() # Also log to console
        ]
    )
    logging.info(f"Logging setup complete. Level: {verbosity.upper()}. Log file: {log_file}")


def get_verbosity(args_verbosity: str) -> str:
    """Gets verbosity level from args or environment variable."""
    env_verbosity = os.getenv("LOG_VERBOSITY", "INFO")
    # Command-line argument takes precedence
    return args_verbosity if args_verbosity else env_verbosity