import sys
import os
import json
from pathlib import Path
import tempfile
import logging
from typing import List, Optional
import ffmpeg
import numpy as np
import librosa
import soundfile as sf
from pydub import AudioSegment
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QListWidget, QProgressBar, QFileDialog,
    QMessageBox, QGroupBox, QRadioButton, QButtonGroup, QCheckBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QMimeData
from PyQt6.QtGui import QDragEnterEvent, QDropEvent

CONFIG_FILE = "audio_processor_config.json"

def setup_logging(output_dir: str):
    # Create a formatter for detailed output
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Create a formatter for console output (simpler)
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    
    # Setup the root logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Create and configure file handler for detailed logging
    log_file = os.path.join(output_dir, 'audio_processor.log')
    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    logger.addHandler(file_handler)
    
    # Create and configure console handler for basic processing info
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    return logger

def load_config():
    default_config = {
        "output_dir": "",
        "slow_audio": True,
        "normalization_methods": []  # List of enabled methods: "peak", "dynamic", "speech"
    }
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                config = json.load(f)
                # Migrate old config format
                if "normalize_audio" in config or "normalization_method" in config:
                    # Convert old config to new format
                    old_normalize = config.get("normalize_audio", True)
                    old_method = config.get("normalization_method", "peak")
                    if old_normalize:
                        config["normalization_methods"] = [old_method]
                    else:
                        config["normalization_methods"] = []
                    # Remove old keys
                    config.pop("normalize_audio", None)
                    config.pop("normalization_method", None)
                # Update with any missing default values
                for key, value in default_config.items():
                    if key not in config:
                        config[key] = value
                return config
        except Exception as e:
            logging.error(f"Error loading config: {e}")
    return default_config

def save_config(config):
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f)
    except Exception as e:
        logging.error(f"Error saving config: {e}")

class AudioProcessor(QThread):
    progress_updated = pyqtSignal(int, str)
    processing_complete = pyqtSignal()
    error_occurred = pyqtSignal(str)

    def __init__(self, input_files: List[str], output_dir: str, slow_down: bool, normalization_methods: List[str]):
        super().__init__()
        self.input_files = input_files
        self.output_dir = output_dir
        self.slow_down = slow_down
        self.normalization_methods = normalization_methods  # List of methods to apply
        self.is_running = True
        logging.debug(f"AudioProcessor initialized with {len(input_files)} files, slow_down={slow_down}, normalization_methods={normalization_methods}")

    def run(self):
        logging.debug("AudioProcessor run method started")
        try:
            total_files = len(self.input_files)
            for i, input_file in enumerate(self.input_files, 1):
                if not self.is_running:
                    logging.debug("Processing stopped by user")
                    break

                try:
                    logging.debug(f"Starting to process file {i}/{total_files}: {input_file}")
                    self.process_file(input_file, i, total_files)
                    logging.debug(f"Successfully processed file {i}/{total_files}")
                except Exception as e:
                    logging.error(f"Error processing {input_file}: {str(e)}", exc_info=True)
                    self.error_occurred.emit(f"Error processing {os.path.basename(input_file)}: {str(e)}")
                    continue

            logging.debug("All files processed, emitting completion signal")
            self.processing_complete.emit()
        except Exception as e:
            logging.error(f"Fatal error in processing thread: {str(e)}", exc_info=True)
            self.error_occurred.emit(f"Fatal error: {str(e)}")

    def process_file(self, input_file: str, current_file: int, total_files: int):
        base_name = os.path.splitext(os.path.basename(input_file))[0]

        logging.info(f"Processing file {current_file}/{total_files}: {base_name}")

        # Determine all output files to create
        output_files = []

        if not self.normalization_methods:
            # No normalization - output un-normalized version(s)
            output_files.append({
                'path': os.path.join(self.output_dir, f"{base_name}.mp3"),
                'method': None,
                'slow': False
            })
            if self.slow_down:
                output_files.append({
                    'path': os.path.join(self.output_dir, f"{base_name}_sl0w.mp3"),
                    'method': None,
                    'slow': True
                })
        else:
            # Create normalized version(s)
            for method in self.normalization_methods:
                suffix_map = {'peak': '_pnorm', 'dynamic': '_dnorm', 'speech': '_snorm'}
                suffix = suffix_map[method]

                # Normal speed version
                output_files.append({
                    'path': os.path.join(self.output_dir, f"{base_name}{suffix}.mp3"),
                    'method': method,
                    'slow': False
                })

                # Slow version if enabled
                if self.slow_down:
                    output_files.append({
                        'path': os.path.join(self.output_dir, f"{base_name}{suffix}_sl0w.mp3"),
                        'method': method,
                        'slow': True
                    })

        # Calculate progress steps
        total_steps = len(output_files) + 1  # +1 for initial extraction
        current_step = 0

        def update_progress(step_description: str):
            nonlocal current_step
            current_step += 1
            progress = int(((current_file - 1) + (current_step / total_steps)) * 100 / total_files)
            self.progress_updated.emit(
                progress,
                f"Processing {base_name} - {step_description} ({current_file}/{total_files})"
            )

        temp_wav_path = None
        temp_norm_path = None
        temp_slow_path = None

        try:
            # Create temporary WAV file
            logging.debug("Creating temporary WAV file")
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_wav:
                temp_wav_path = temp_wav.name

            update_progress("Extracting audio")
            logging.info("Extracting audio...")
            try:
                version_output = ffmpeg.probe(input_file)
                logging.debug(f"FFmpeg probe result: {version_output}")
            except ffmpeg.Error as e:
                error_msg = f"FFmpeg probe error: {str(e)}"
                if e.stderr:
                    error_msg += f"\nFFmpeg stderr: {e.stderr.decode()}"
                logging.error(error_msg)
                raise Exception(f"Failed to analyze audio file: {str(e)}")

            try:
                logging.debug(f"Starting FFmpeg extraction to {temp_wav_path}")
                stream = ffmpeg.input(input_file)
                stream = ffmpeg.output(stream, temp_wav_path, acodec='pcm_s16le', ac=2, loglevel='info')
                
                cmd = ffmpeg.compile(stream, overwrite_output=True)
                logging.debug(f"FFmpeg command: {' '.join(cmd)}")
                
                out, err = ffmpeg.run(stream, capture_stdout=True, capture_stderr=True, overwrite_output=True)
                logging.debug("FFmpeg extraction completed successfully")
                if err:
                    logging.debug(f"FFmpeg stderr (info): {err.decode()}")
            except ffmpeg.Error as e:
                error_msg = "FFmpeg extraction failed"
                if e.stderr:
                    error_msg += f"\nFFmpeg stderr: {e.stderr.decode()}"
                logging.error(error_msg)
                raise Exception(f"Failed to extract audio: {str(e)}")

            # Process each output file
            for output_info in output_files:
                output_path = output_info['path']
                method = output_info['method']
                is_slow = output_info['slow']

                desc = f"Creating {os.path.basename(output_path)}"
                update_progress(desc)

                if method == "speech":
                    # Speech normalization: use FFmpeg directly to avoid WAV size limitations
                    logging.info(f"Creating {os.path.basename(output_path)} with speech normalization...")
                    try:
                        stream = ffmpeg.input(temp_wav_path)

                        if is_slow:
                            # Apply both speech normalization and tempo change
                            stream = ffmpeg.output(
                                stream,
                                output_path,
                                af='loudnorm,speechnorm,loudnorm,atempo=0.6',
                                acodec='libmp3lame',
                                ar=44100,
                                ab='320k',
                                loglevel='info'
                            )
                        else:
                            # Apply only speech normalization
                            stream = ffmpeg.output(
                                stream,
                                output_path,
                                af='loudnorm,speechnorm,loudnorm',
                                acodec='libmp3lame',
                                ar=44100,
                                ab='320k',
                                loglevel='info'
                            )

                        ffmpeg.run(stream, capture_stdout=True, capture_stderr=True, overwrite_output=True)
                        logging.info(f"Successfully created: {output_path}")

                    except ffmpeg.Error as e:
                        error_msg = "FFmpeg speech normalization failed"
                        if e.stderr:
                            error_msg += f"\nFFmpeg stderr: {e.stderr.decode()}"
                        logging.error(error_msg)
                        raise Exception(f"Failed to create {os.path.basename(output_path)}: {str(e)}")

                elif method == "dynamic":
                    # Dynamic normalization: use FFmpeg dynaudnorm
                    logging.info(f"Creating {os.path.basename(output_path)} with dynamic normalization...")
                    try:
                        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_norm:
                            temp_norm_path = temp_norm.name

                        stream = ffmpeg.input(temp_wav_path)
                        stream = ffmpeg.output(stream, temp_norm_path, af='dynaudnorm', loglevel='info')
                        ffmpeg.run(stream, capture_stdout=True, capture_stderr=True, overwrite_output=True)

                        # Load normalized audio and export
                        audio = AudioSegment.from_wav(temp_norm_path)

                        if is_slow:
                            # Create slowed version
                            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_slow:
                                temp_slow_path = temp_slow.name

                            y, sr = librosa.load(temp_norm_path, sr=None, mono=False)
                            if y.ndim == 1:
                                y_stretched = librosa.effects.time_stretch(y, rate=0.6)
                            else:
                                y_stretched = np.array([
                                    librosa.effects.time_stretch(y[0], rate=0.6),
                                    librosa.effects.time_stretch(y[1], rate=0.6)
                                ])

                            if y_stretched.ndim == 1:
                                sf.write(temp_slow_path, y_stretched, sr)
                            else:
                                sf.write(temp_slow_path, y_stretched.T, sr)

                            slow_audio = AudioSegment.from_wav(temp_slow_path)
                            slow_audio.export(output_path, format='mp3')

                            if os.path.exists(temp_slow_path):
                                os.unlink(temp_slow_path)
                        else:
                            # Export normal speed
                            audio.export(output_path, format='mp3')

                        if os.path.exists(temp_norm_path):
                            os.unlink(temp_norm_path)

                        logging.info(f"Successfully created: {output_path}")

                    except Exception as e:
                        raise Exception(f"Failed to create {os.path.basename(output_path)}: {str(e)}")

                elif method == "peak":
                    # Peak normalization: use pydub
                    logging.info(f"Creating {os.path.basename(output_path)} with peak normalization...")
                    try:
                        # Load audio from original WAV
                        audio = AudioSegment.from_wav(temp_wav_path)

                        # Apply peak normalization
                        peak_dbfs = audio.max_dBFS
                        target_peak_dbfs = -1.0
                        change_in_dbfs = target_peak_dbfs - peak_dbfs
                        audio = audio.apply_gain(change_in_dbfs)
                        logging.debug(f"Peak normalization: adjusted from {peak_dbfs:.2f} dBFS to {target_peak_dbfs:.2f} dBFS")

                        if is_slow:
                            # Create slowed version
                            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_norm:
                                temp_norm_path = temp_norm.name
                            audio.export(temp_norm_path, format='wav')

                            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_slow:
                                temp_slow_path = temp_slow.name

                            y, sr = librosa.load(temp_norm_path, sr=None, mono=False)
                            if y.ndim == 1:
                                y_stretched = librosa.effects.time_stretch(y, rate=0.6)
                            else:
                                y_stretched = np.array([
                                    librosa.effects.time_stretch(y[0], rate=0.6),
                                    librosa.effects.time_stretch(y[1], rate=0.6)
                                ])

                            if y_stretched.ndim == 1:
                                sf.write(temp_slow_path, y_stretched, sr)
                            else:
                                sf.write(temp_slow_path, y_stretched.T, sr)

                            slow_audio = AudioSegment.from_wav(temp_slow_path)
                            slow_audio.export(output_path, format='mp3')

                            if os.path.exists(temp_slow_path):
                                os.unlink(temp_slow_path)
                            if os.path.exists(temp_norm_path):
                                os.unlink(temp_norm_path)
                        else:
                            # Export normal speed
                            audio.export(output_path, format='mp3')

                        logging.info(f"Successfully created: {output_path}")

                    except Exception as e:
                        raise Exception(f"Failed to create {os.path.basename(output_path)}: {str(e)}")

                else:
                    # No normalization - just convert to MP3
                    logging.info(f"Creating {os.path.basename(output_path)} without normalization...")
                    try:
                        if is_slow:
                            # Create slowed version from original WAV
                            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_slow:
                                temp_slow_path = temp_slow.name

                            y, sr = librosa.load(temp_wav_path, sr=None, mono=False)
                            if y.ndim == 1:
                                y_stretched = librosa.effects.time_stretch(y, rate=0.6)
                            else:
                                y_stretched = np.array([
                                    librosa.effects.time_stretch(y[0], rate=0.6),
                                    librosa.effects.time_stretch(y[1], rate=0.6)
                                ])

                            if y_stretched.ndim == 1:
                                sf.write(temp_slow_path, y_stretched, sr)
                            else:
                                sf.write(temp_slow_path, y_stretched.T, sr)

                            slow_audio = AudioSegment.from_wav(temp_slow_path)
                            slow_audio.export(output_path, format='mp3')

                            if os.path.exists(temp_slow_path):
                                os.unlink(temp_slow_path)
                        else:
                            # Export normal speed from original WAV
                            audio = AudioSegment.from_wav(temp_wav_path)
                            audio.export(output_path, format='mp3')

                        logging.info(f"Successfully created: {output_path}")

                    except Exception as e:
                        raise Exception(f"Failed to create {os.path.basename(output_path)}: {str(e)}")

            logging.info(f"Successfully processed: {base_name}")
            self.progress_updated.emit(
                int(current_file * 100 / total_files),
                f"Completed {base_name} ({current_file}/{total_files})"
            )

        except Exception as e:
            error_msg = f"Error processing {base_name}: {str(e)}"
            logging.error(error_msg, exc_info=True)
            self.error_occurred.emit(error_msg)
            raise e

        finally:
            # Clean up temporary files
            logging.debug("Cleaning up temporary files")
            for temp_file, file_type in [
                (temp_wav_path, "WAV"),
                (temp_norm_path, "normalized"),
                (temp_slow_path, "slowed")
            ]:
                try:
                    if temp_file and os.path.exists(temp_file):
                        os.unlink(temp_file)
                        logging.debug(f"Removed temporary {file_type} file: {temp_file}")
                except Exception as e:
                    logging.warning(f"Failed to remove temporary {file_type} file: {str(e)}")

    def stop(self):
        self.is_running = False


class DropArea(QWidget):
    files_dropped = pyqtSignal(list)

    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)
        layout = QVBoxLayout()
        self.label = QLabel("Drop video files here or use 'Add Files' button")
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.label)
        self.setLayout(layout)
        self.setMinimumHeight(100)
        self.setStyleSheet("""
            QWidget {
                border: 2px dashed #aaa;
                border-radius: 5px;
                background-color: #f0f0f0;
            }
            QLabel {
                color: #333333;
            }
        """)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        files = [url.toLocalFile() for url in event.mimeData().urls()]
        self.files_dropped.emit(files)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Audio Processor")
        self.setMinimumSize(600, 450)  # Increased height for new controls
        self.processor: Optional[AudioProcessor] = None
        
        # Load config
        self.config = load_config()
        self.setup_ui()

    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Drop area
        self.drop_area = DropArea()
        self.drop_area.files_dropped.connect(self.add_files)
        layout.addWidget(self.drop_area)

        # File list
        self.file_list = QListWidget()
        layout.addWidget(self.file_list)

        # Output directory selection
        output_layout = QHBoxLayout()
        self.output_label = QLabel("Output Directory:")
        self.output_path = QLabel(self.config["output_dir"] or "Not selected")
        self.output_path.setStyleSheet("""
            QLabel {
                padding: 5px;
                background-color: #f0f0f0;
                border: 1px solid #ccc;
                color: #333333;
            }
        """)
        self.output_button = QPushButton("Select Output Folder")
        self.output_button.clicked.connect(self.select_output_directory)
        
        output_layout.addWidget(self.output_label)
        output_layout.addWidget(self.output_path, stretch=1)
        output_layout.addWidget(self.output_button)
        layout.addLayout(output_layout)

        # Processing options
        options_layout = QVBoxLayout()

        # Normalization options
        norm_group = QGroupBox("Audio Normalization (select none or more)")
        norm_layout = QVBoxLayout()

        # Normalization method checkboxes
        self.norm_peak = QCheckBox("Peak Normalization")
        self.norm_dynamic = QCheckBox("Dynamic Normalization")
        self.norm_speech = QCheckBox("Speech Normalization")

        # Set checked state from config
        self.norm_peak.setChecked("peak" in self.config["normalization_methods"])
        self.norm_dynamic.setChecked("dynamic" in self.config["normalization_methods"])
        self.norm_speech.setChecked("speech" in self.config["normalization_methods"])

        norm_layout.addWidget(self.norm_peak)
        norm_layout.addWidget(self.norm_dynamic)
        norm_layout.addWidget(self.norm_speech)

        norm_group.setLayout(norm_layout)

        # Tempo options
        tempo_group = QGroupBox("Slow Audio (60%)")
        tempo_layout = QVBoxLayout()
        self.tempo_on = QRadioButton("On")
        self.tempo_off = QRadioButton("Off")
        self.tempo_on.setChecked(self.config["slow_audio"])
        self.tempo_off.setChecked(not self.config["slow_audio"])
        tempo_layout.addWidget(self.tempo_on)
        tempo_layout.addWidget(self.tempo_off)
        tempo_group.setLayout(tempo_layout)

        options_layout.addWidget(norm_group)
        options_layout.addWidget(tempo_group)
        layout.addLayout(options_layout)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        layout.addWidget(self.progress_bar)

        # Buttons
        button_layout = QHBoxLayout()
        self.add_button = QPushButton("Add Files")
        self.add_button.clicked.connect(self.show_file_dialog)
        self.process_button = QPushButton("Process Files")
        self.process_button.clicked.connect(self.process_files)
        self.clear_button = QPushButton("Clear List")
        self.clear_button.clicked.connect(self.clear_files)
        
        button_layout.addWidget(self.add_button)
        button_layout.addWidget(self.process_button)
        button_layout.addWidget(self.clear_button)
        layout.addLayout(button_layout)

        # Create button group for tempo radio buttons
        self.tempo_group = QButtonGroup()
        self.tempo_group.addButton(self.tempo_on)
        self.tempo_group.addButton(self.tempo_off)

        # Connect signals
        self.tempo_on.toggled.connect(self.save_options)
        self.norm_peak.toggled.connect(self.save_options)
        self.norm_dynamic.toggled.connect(self.save_options)
        self.norm_speech.toggled.connect(self.save_options)

    def save_options(self):
        self.config["slow_audio"] = self.tempo_on.isChecked()

        # Collect all checked normalization methods
        methods = []
        if self.norm_peak.isChecked():
            methods.append("peak")
        if self.norm_dynamic.isChecked():
            methods.append("dynamic")
        if self.norm_speech.isChecked():
            methods.append("speech")

        self.config["normalization_methods"] = methods
        save_config(self.config)

    def select_output_directory(self):
        dir_path = QFileDialog.getExistingDirectory(
            self,
            "Select Output Directory",
            self.config["output_dir"] or ""
        )
        if dir_path:
            self.config["output_dir"] = dir_path
            self.output_path.setText(dir_path)
            save_config(self.config)

    def show_file_dialog(self):
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Video Files",
            "",
            "Video Files (*.mp4 *.avi *.mkv *.mov *.wmv);;All Files (*.*)"
        )
        self.add_files(files)

    def add_files(self, files: List[str]):
        for file in files:
            if os.path.isfile(file) and not self.file_exists(file):
                self.file_list.addItem(file)

    def file_exists(self, file_path: str) -> bool:
        for i in range(self.file_list.count()):
            if self.file_list.item(i).text() == file_path:
                return True
        return False

    def clear_files(self):
        self.file_list.clear()

    def process_files(self):
        if self.file_list.count() == 0:
            QMessageBox.warning(self, "No Files", "Please add some files to process.")
            return

        if not self.config["output_dir"]:
            QMessageBox.warning(self, "No Output Directory", "Please select an output directory.")
            return

        if not os.path.exists(self.config["output_dir"]):
            try:
                os.makedirs(self.config["output_dir"])
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Could not create output directory: {str(e)}")
                return

        # Setup logging for this processing session
        logger = setup_logging(self.config["output_dir"])
        logger.info("=== Starting new audio processing session ===")
        logger.info(f"Output directory: {self.config['output_dir']}")

        # Collect selected normalization methods
        normalization_methods = []
        if self.norm_peak.isChecked():
            normalization_methods.append("peak")
        if self.norm_dynamic.isChecked():
            normalization_methods.append("dynamic")
        if self.norm_speech.isChecked():
            normalization_methods.append("speech")

        logger.info(f"Normalization methods: {normalization_methods if normalization_methods else 'None'}")
        logger.info(f"Slow audio: {'ON' if self.tempo_on.isChecked() else 'OFF'}")

        if self.processor and self.processor.isRunning():
            logger.info("Stopping current processing...")
            self.processor.stop()
            self.processor.wait()
            self.process_button.setText("Process Files")
            self.enable_controls(True)
        else:
            files = [self.file_list.item(i).text() for i in range(self.file_list.count())]
            logger.info(f"Found {len(files)} files to process")

            self.processor = AudioProcessor(
                files,
                self.config["output_dir"],
                self.tempo_on.isChecked(),
                normalization_methods
            )
            self.processor.progress_updated.connect(self.update_progress)
            self.processor.processing_complete.connect(self.processing_finished)
            self.processor.error_occurred.connect(self.show_error)
            self.processor.start()
            
            self.process_button.setText("Stop Processing")
            self.enable_controls(False)

    def enable_controls(self, enabled: bool):
        logging.debug(f"Setting controls enabled: {enabled}")
        self.add_button.setEnabled(enabled)
        self.clear_button.setEnabled(enabled)
        self.output_button.setEnabled(enabled)
        self.norm_peak.setEnabled(enabled)
        self.norm_dynamic.setEnabled(enabled)
        self.norm_speech.setEnabled(enabled)
        self.tempo_on.setEnabled(enabled)
        self.tempo_off.setEnabled(enabled)

    def update_progress(self, value: int, message: str):
        self.progress_bar.setValue(value)
        self.progress_bar.setFormat(f"{message} - {value}%")

    def processing_finished(self):
        self.progress_bar.setValue(100)
        self.progress_bar.setFormat("Processing complete!")
        self.process_button.setText("Process Files")
        self.enable_controls(True)
        logging.info("Processing completed successfully!")
        QMessageBox.information(self, "Complete", "All files have been processed successfully!")

    def show_error(self, message: str):
        QMessageBox.warning(self, "Error", message)

    def closeEvent(self, event):
        if self.processor and self.processor.isRunning():
            self.processor.stop()
            self.processor.wait()
        event.accept()


if __name__ == '__main__':
    logging.debug("Application starting")
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    logging.debug("Main window displayed")
    sys.exit(app.exec()) 