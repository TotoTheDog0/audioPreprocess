# Audio Processor

A PyQt6-based desktop application for batch processing audio from video files with multiple normalization options and tempo control.

## Purpose

Audio Processor is a GUI tool designed to extract and process audio from video files for subtitling workflows. It provides:

- **Audio Extraction**: Extract audio tracks from various video formats (MP4, AVI, MKV, MOV, WMV and possibly others)
- **Multiple Normalization Methods**:
  - Peak normalization (adjusts audio to maximum level without distortion)
  - Dynamic normalization (evens out volume variations)
  - Speech normalization (optimized for spoken content)
- **Tempo Control**: Create slowed-down versions at 60% speed for better transcription
- **Batch Processing**: Process multiple files in a single operation
- **Drag-and-Drop Interface**: Simple, intuitive file management

## Features

- Modern PyQt6 graphical user interface
- Drag-and-drop file support
- Multiple normalization methods (can apply one, multiple, or none)
- Optional 60% speed reduction for language learning
- Progress tracking with detailed status messages
- Comprehensive logging (saved to output directory)
- Configuration persistence (remembers your last settings)
- High-quality MP3 output (320 kbps)

## System Requirements

- Python 3.8 or higher
- FFmpeg (must be installed separately and available in system PATH)
- Windows, macOS, or Linux

## Installation

### Installing FFmpeg

Before installing the Python dependencies, you must install FFmpeg:

**Windows:**
1. Download FFmpeg from [ffmpeg.org](https://ffmpeg.org/download.html)
2. Extract to a folder (e.g., `C:\ffmpeg`)
3. Add the `bin` folder to your system PATH environment variable

**macOS:**
```bash
brew install ffmpeg
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install ffmpeg
```

### Option 1: Installation with Virtual Environment (Recommended)

Using a virtual environment isolates the project dependencies from your system Python installation.

**Windows:**
```bash
# Navigate to the project directory
cd "\path\to\audioPreprocess"

# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**macOS/Linux:**
```bash
# Navigate to the project directory
cd "/path/to/audioPreprocess"

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Option 2: Installation without Virtual Environment

Install dependencies directly to your system Python:

```bash
# Navigate to the project directory
cd "\path\to\audioPreprocess"

# Install dependencies
pip install PyQt6 ffmpeg-python numpy librosa soundfile pydub
```

### Creating requirements.txt (if not present)

If you don't have a requirements.txt file, create one with these dependencies:

```
PyQt6>=6.4.0
ffmpeg-python>=0.2.0
numpy>=1.24.0
librosa>=0.10.0
soundfile>=0.12.0
pydub>=0.25.0
```

Save this as `requirements.txt` in the project directory.

## Usage Guide

### Starting the Application

**With Virtual Environment:**
```bash
# Activate the virtual environment first (if not already activated)
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Run the application
python audio_processor.py
```

**Without Virtual Environment:**
```bash
python audio_processor.py
```

### Using the Application

1. **Select Output Directory**
   - Click "Select Output Folder" to choose where processed files will be saved
   - This setting is remembered for future sessions

2. **Add Video Files**
   - Click "Add Files" and select video files, OR
   - Drag and drop video files directly onto the drop area
   - Supported formats: MP4, AVI, MKV, MOV, WMV (possibly others)

3. **Configure Processing Options**

   **Normalization Options** (can select multiple or none):
   - **Peak Normalization**: Adjusts audio to maximum level without clipping (fastest)
   - **Dynamic Normalization**: Evens out volume variations across the file
   - **Speech Normalization**: Optimizes for spoken content with advanced filters

   **Slow Audio**:
   - **On**: Creates slowed versions at 60% speed (files will have `_sl0w` suffix)
   - **Off**: Processes at normal speed only

4. **Process Files**
   - Click "Process Files" to start batch processing
   - Progress bar shows current file and overall progress
   - Click "Stop Processing" to cancel the operation

5. **Review Results**
   - Processed MP3 files are saved to the output directory
   - A log file (`audio_processor.log`) is created in the output directory
   - Check the log file for detailed processing information

### Output File Naming

Files are named based on the original filename and selected options:

- `filename.mp3` - No normalization, normal speed
- `filename_pnorm.mp3` - Peak normalization, normal speed
- `filename_dnorm.mp3` - Dynamic normalization, normal speed
- `filename_snorm.mp3` - Speech normalization, normal speed
- `filename_pnorm_sl0w.mp3` - Peak normalization, 60% speed
- `filename_dnorm_sl0w.mp3` - Dynamic normalization, 60% speed
- `filename_snorm_sl0w.mp3` - Speech normalization, 60% speed

If multiple normalization methods are selected, separate files are created for each method.

### Configuration File

The application stores settings in `audio_processor_config.json` in the application directory:
- Last used output directory
- Normalization method selections
- Tempo preference

You can manually edit this file if needed, or delete it to reset to defaults.

### Logging

Each processing session creates detailed logs in `audio_processor.log` in the output directory:
- Timestamp for each operation
- Processing steps and progress
- Error messages and stack traces
- FFmpeg command output (debug level)

Check this file if you encounter issues during processing.

## Troubleshooting

**FFmpeg not found error:**
- Ensure FFmpeg is installed and available in your system PATH
- Test by running `ffmpeg -version` in your terminal/command prompt

**Audio extraction fails:**
- Verify the video file is not corrupted
- Check that the video file contains an audio track
- Review the log file for detailed error messages

**Slow processing with speech normalization:**
- Speech normalization is computationally intensive
- This is normal behavior for high-quality audio processing
- Consider using peak or dynamic normalization for faster results

**Memory errors with large files:**
- Process fewer files at once
- Close other applications to free up memory
- Use peak normalization which is less memory-intensive

## Technical Details

- **Audio Format**: MP3, 320 kbps, 44.1 or 48 kHz
- **Channels**: Stereo (2 channels)
- **Speed Reduction**: 60% using librosa time-stretching (pitch-preserved)
- **Normalization Targets**:
  - Peak: -1.0 dBFS
  - Dynamic: FFmpeg dynaudnorm filter
  - Speech: FFmpeg loudnorm + speechnorm + loudnorm chain

## License

Copyright 2025

This project is licensed under the **Apache License 2.0**.

### What this means:
- ✅ You can use this software for any purpose (commercial or non-commercial)
- ✅ You can modify the source code
- ✅ You can distribute the software
- ✅ You can distribute modified versions
- ✅ You can sublicense (include in proprietary software)
- ⚠️ You must include the license and copyright notice
- ⚠️ You must state significant changes made to the code
- ⚠️ You must include the NOTICE file if one exists
- ❌ The software is provided "as is" with no warranty
- ❌ The authors are not liable for any damages

See the full license text at: http://www.apache.org/licenses/LICENSE-2.0

## Support

For issues, questions, or contributions, please refer to the project documentation or contact the maintainer.

## Credits

Built with:
- [PyQt6](https://www.riverbankcomputing.com/software/pyqt/) - GUI framework
- [FFmpeg](https://ffmpeg.org/) - Audio/video processing
- [librosa](https://librosa.org/) - Audio analysis and time-stretching
- [pydub](https://github.com/jiaaro/pydub) - Audio manipulation
- [NumPy](https://numpy.org/) - Numerical computing
- [soundfile](https://python-soundfile.readthedocs.io/) - Audio file I/O
