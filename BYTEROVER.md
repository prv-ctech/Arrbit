# Byterover Handbook

_Generated: September 5, 2025_

## Layer 1: System Overview

**Purpose**: Arrbit is a multimedia automation and processing framework designed for \*arr applications (Lidarr, Sonarr, Radarr) and Tdarr. It provides automated configuration, custom scripts, and AI-enhanced plugins for media management and transcoding workflows.

**Tech Stack**:

- **Core**: Bash scripting for automation and configuration
- **AI Processing**: Python with WhisperX, Lingua language detection
- **Media Processing**: FFmpeg, mkvpropedit for container manipulation
- **Plugin System**: JavaScript/Node.js for Tdarr plugins
- **Container Runtime**: Docker/Unraid integration

**Architecture**: **Plugin-based Automation Architecture**

- Modular plugin system for Tdarr media processing
- Service-based configuration for \*arr applications
- Universal connectors for cross-application integration
- Environment isolation using Python virtual environments

**Key Technical Decisions**:

- CPU-optimized AI processing (int8 compute, tiny models)
- Multi-sample audio analysis for robust language detection
- Direct container manipulation to avoid re-encoding when possible
- Comprehensive logging and debugging infrastructure

**Entry Points**:

- Tdarr plugins: `tdarr/scripts/plugins/`
- Lidarr automation: `lidarr/scripts/setup/setup.bash`
- Universal helpers: `universal/helpers/`

---

## Layer 2: Module Map

**Core Modules**:

- **Tdarr Plugins**: AI-enhanced media processing plugins for language detection, subtitle management, and audio optimization
- **Lidarr Integration**: Automated configuration modules for quality profiles, metadata management, and custom formats
- **Universal Connectors**: Cross-application bridge utilities and shared helper functions

**Data Layer**:

- **Configuration Templates**: JSON payloads for automated service configuration
- **Reference Data**: Genre whitelists, quality definitions, and metadata profiles
- **Temporary Processing**: AI model outputs, logs, and intermediate processing files

**Integration Points**:

- **Arr Bridge**: Universal connector for \*arr application APIs
- **Python AI Environment**: Isolated environment for WhisperX and Lingua processing
- **Container Runtime**: Docker/Unraid integration points

**Utilities**:

- **Logging Utils**: Centralized logging and debugging infrastructure
- **Helper Functions**: Shared utilities for path manipulation, error handling
- **Test Framework**: Testing utilities for plugin and script validation

**Module Dependencies**:

- Tdarr plugins → Universal helpers → Python AI environment
- Lidarr automation → Universal connectors → Arr APIs
- All modules → Logging utilities for debugging

---

## Layer 3: Integration Guide

**API Endpoints**:

- **Lidarr API**: Quality profiles, metadata profiles, custom formats management
- **Tdarr Processing API**: Plugin execution and status monitoring
- **FFmpeg Integration**: Media analysis and processing commands

**Configuration Files**:

- `lidarr/config/arrbit-config.conf`: Main configuration settings
- `lidarr/config/beets-config.yaml`: Beets integration settings
- `lidarr/data/payload-*.json`: Service configuration templates

**External Integrations**:

- **WhisperX**: CPU-optimized speech recognition with VAD preprocessing
- **Lingua Language Detector**: High-accuracy language detection with confidence scoring
- **mkvpropedit**: Direct MKV container manipulation without re-encoding
- **FFmpeg**: Audio/video analysis and processing

**Workflows**:

1. **AI Language Detection**: Multi-sample audio extraction → WhisperX transcription → Lingua detection → Consensus voting
2. **Lidarr Setup**: Dependencies → Configuration → API payload application → Custom scripts deployment
3. **Media Processing**: Plugin trigger → Analysis → Processing → Metadata application

**Interface Definitions**:

- Tdarr Plugin Interface: `details()` and `plugin(file, librarySettings, inputs, otherArguments)`
- Bash Module Interface: Sourced functions with standardized error handling
- Python AI Interface: JSON input/output with comprehensive logging

---

## Layer 4: Extension Points

**Design Patterns**:

- **Plugin Architecture**: Extensible Tdarr plugin system with standardized interfaces
- **Configuration Templates**: JSON-based payloads for service automation
- **Environment Isolation**: Python virtual environments for AI processing dependencies
- **Logging Pipeline**: Centralized debugging with file and stderr output

**Extension Points**:

- **New AI Models**: Plugin template supports any WhisperX-compatible model
- **Additional Languages**: Lingua detector includes all spoken languages
- **Custom Tdarr Plugins**: Template structure for new media processing plugins
- **Service Integrations**: Universal connector pattern for new \*arr applications

**Customization Areas**:

- **AI Processing Parameters**: Confidence thresholds, sample counts, model selection
- **Quality Profiles**: Customizable codec preferences and quality settings
- **Metadata Handling**: Extensible metadata extraction and application workflows
- **Error Handling**: Configurable timeout values and retry logic

**Plugin Architecture**:

- **Tdarr Plugin System**: JavaScript-based with standardized input/output interfaces
- **Bash Module System**: Sourceable modules with shared helper functions
- **Python AI Framework**: Isolated environments with dependency management
- **Configuration Pipeline**: Template-based automation with payload injection

**Recent Changes**:

- Enhanced multi-sample language detection with majority voting
- CPU-optimized AI processing with int8 compute type
- Direct MKV manipulation using mkvpropedit
- Comprehensive logging and debugging infrastructure

---

_Byterover handbook optimized for agent navigation and human developer onboarding_
