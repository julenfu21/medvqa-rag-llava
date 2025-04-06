import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.utils.data_definitions import LoggerConfig
from src.utils.enums import LogLevel


class LoggerManager:

    def __init__(
        self,
        log_save_directory: Path,
        logger_name: str = "VisualQALogger",
        logger_config: LoggerConfig = LoggerConfig(
            console_handler_enabled=True,
            file_handler_enabled=True
        )
    ) -> None:
        self.__log_save_directory = log_save_directory
        self.__log_filename = None
        self.__logger_name = logger_name
        self.__logger_config = logger_config
        self.__logger_formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] - %(message)s"
        )
        self.__logger = logging.getLogger(self.__logger_name)
        self.__setup_logger()

    @property
    def log_filename(self) -> str:
        return self.__log_filename

    def __setup_logger(self) -> None:
        self.__logger.setLevel(logging.DEBUG)

        if not self.__logger.handlers:
            if self.__logger_config.console_handler_enabled:
                console_handler = logging.StreamHandler()
                console_handler.setFormatter(self.__logger_formatter)
                self.__logger.addHandler(console_handler)

            if self.__logger_config.file_handler_enabled:
                self.__log_save_directory.mkdir(parents=True, exist_ok=True)

            print(f"Logger '{self.__logger_name}' created!")
            self.__print_logger_info()
        else:
            print(f"Logger '{self.__logger_name}' already exists. Reusing existing logger.")
            self.__print_logger_info()

    def create_new_log_file(self, log_filename: Optional[str] = None) -> None:
        if not self.__logger_config.file_handler_enabled:
            print("File handler is disabled. Log files cannot be created!")
            return

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if log_filename:
            self.__log_filename = f"{timestamp}_{log_filename}"
        else:
            self.__log_filename = f"visual-qa_{timestamp}.log"

        old_handler = None
        for handler in self.__logger.handlers:
            if isinstance(handler, logging.FileHandler):
                old_handler = handler
                break
        if old_handler:
            self.__logger.removeHandler(old_handler)
            old_handler.close()

        file_handler = logging.FileHandler(self.__log_save_directory / self.__log_filename)
        file_handler.setFormatter(self.__logger_formatter)
        self.__logger.addHandler(file_handler)

        print(f"New log file created for the '{self.__logger_name}' logger!")
        self.__print_logger_info()

    def log(self, level: LogLevel, message: str) -> None:
        log_filepath = Path(self.__log_save_directory / self.__log_filename)
        if not log_filepath.exists():
            print("The log file might have been accidentally deleted or not created!")
            self.create_new_log_file()

        self.__logger.log(level.value, message)

    def __print_logger_info(self) -> None:

        def handler_status(handler_option: bool) -> str:
            if handler_option:
                return "Enabled"
            return "Disabled"

        print((
            f"\t- Log Directory: {self.__log_save_directory}\n"
            f"\t- Log Filename: {self.__log_filename}\n"
            "\t- Handlers:\n"
            "\t\t* Console Handler: "
            f"{handler_status(self.__logger_config.console_handler_enabled)}\n"
            f"\t\t* File Handler: {handler_status(self.__logger_config.file_handler_enabled)}"
        ), end="\n\n")
