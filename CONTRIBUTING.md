# Contributing to YOLO Zone Detection

Thank you for your interest in contributing to the YOLO Zone Detection System! 🎉

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
- [Development Setup](#development-setup)
- [Coding Standards](#coding-standards)
- [Submitting Changes](#submitting-changes)

---

## Code of Conduct

This project adheres to a code of conduct that all contributors are expected to follow. By participating, you are expected to uphold this standard. Please be respectful and constructive in all interactions.

---

## How Can I Contribute?

### 🐛 Reporting Bugs

Before creating bug reports, please check existing issues. When creating a bug report, include:

- **Clear title** and description
- **Steps to reproduce** the issue
- **Expected behavior** vs actual behavior
- **System information**: OS, Python version, camera type
- **Logs and error messages** (if applicable)
- **Screenshots or videos** (if relevant)

### 💡 Suggesting Enhancements

Enhancement suggestions are welcome! Include:

- **Clear use case** - why is this needed?
- **Proposed solution** - how should it work?
- **Alternatives considered** - what other approaches did you think about?

### 🔧 Code Contributions

1. **Bug Fixes**: Always welcome
2. **New Features**: Discuss in an issue first
3. **Documentation**: Corrections and improvements
4. **Performance**: Optimization proposals

---

## Development Setup

### 1. Fork and Clone

```bash
git clone https://github.com/YOUR-USERNAME/yolo-zone-detection.git
cd yolo-zone-detection
```

### 2. Create Development Environment

```bash
python3.10 -m venv venv
source venv/bin/activate  # or .\venv\Scripts\activate on Windows
pip install -r requirements.txt
pip install -r requirements-dev.txt  # if exists
```

### 3. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/issue-number-description
```

---

## Coding Standards

### Python Style

- Follow **PEP 8** style guide
- Use **type hints** where appropriate
- Write **docstrings** for all functions/classes
- Keep functions focused and < 50 lines when possible

### Code Structure

```python
"""
Module description.

Detailed explanation of what this module does.
"""

import standard_library
import third_party
from local_module import something


class MyClass:
    """One-line class description.
    
    Detailed explanation of the class purpose and usage.
    
    Attributes:
        attr_name (type): Description of attribute.
    """
    
    def __init__(self, param: str):
        """Initialize the class.
        
        Args:
            param: Description of parameter.
        """
        self.attr_name = param
    
    def method_name(self, arg: int) -> bool:
        """One-line method description.
        
        Detailed explanation if needed.
        
        Args:
            arg: Description of argument.
            
        Returns:
            Description of return value.
            
        Raises:
            ValueError: When something is wrong.
        """
        return True
```

### Naming Conventions

- **Classes**: `PascalCase` (e.g., `ZoneDetectionApp`)
- **Functions/Methods**: `snake_case` (e.g., `process_frame`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `MAX_FPS`)
- **Private methods**: `_leading_underscore` (e.g., `_internal_method`)

### Comments

- Write self-documenting code
- Add comments for complex logic
- Use TODO comments sparingly: `# TODO(username): Description`
- Avoid obvious comments

```python
# ❌ Bad
x = x + 1  # Increment x

# ✅ Good
# Compensate for border width in zone calculation
zone_width = frame_width - (2 * border_width)
```

---

## Testing

### Running Tests

```bash
# Find cameras (integration test)
python -m tools.find_cameras

# Test MQTT connection
python -m tools.mqtt_subscriber

# Test main detection (stop with Ctrl+C after few seconds)
python -m src.main --camera 0
```

### Before Submitting

- [ ] Code runs without errors
- [ ] All modified files follow coding standards
- [ ] Documentation updated (if applicable)
- [ ] CHANGELOG.md updated (if adding features)

---

## Submitting Changes

### Pull Request Process

1. **Update Documentation**
   - Update README.md if adding features
   - Add docstrings to new functions/classes
   - Update QUICKSTART.md if changing setup

2. **Commit Messages**
   ```
   Short summary (50 chars or less)
   
   More detailed explanation if needed. Wrap at 72 characters.
   Include motivation for the change and contrast with previous behavior.
   
   - Bullet points are fine
   - Use present tense: "Add feature" not "Added feature"
   - Reference issues: "Fixes #123" or "Relates to #456"
   ```

3. **Create Pull Request**
   - Clear title describing the change
   - Reference related issues
   - Describe what changed and why
   - Include screenshots/videos if UI changed
   - List any breaking changes

4. **Review Process**
   - Maintainers will review your PR
   - Address feedback constructively
   - Update PR as needed
   - Once approved, it will be merged

### Commit Message Examples

```bash
# Good commits
git commit -m "Add RTSP camera support to config manager"
git commit -m "Fix FPS calculation in performance monitor"
git commit -m "Update README with Docker installation steps"

# Bad commits (avoid these)
git commit -m "Update"
git commit -m "Fix bug"
git commit -m "Changes"
```

---

## Project Structure

Understanding the codebase:

```
src/
├── main.py          # Application entry point and main loop
├── config.py        # Configuration management
├── camera.py        # Camera capture and management
├── detector.py      # YOLO detection and tracking
├── mqtt_client.py   # MQTT event publishing
├── performance.py   # FPS and performance monitoring
└── web_dashboard.py # Optional web interface

tools/
├── find_cameras.py      # Camera discovery utility
└── mqtt_subscriber.py   # MQTT event monitor

scripts/
├── export.py       # Model export to OpenVINO
└── setup.py        # Automated setup
```

---

## Areas Needing Help

Looking for contributions in these areas:

- 📝 **Documentation**: Tutorials, examples, translations
- 🧪 **Testing**: Unit tests, integration tests
- 🚀 **Performance**: Optimization, profiling
- 🌐 **Features**: New camera types, detection algorithms
- 🐛 **Bug Fixes**: Check open issues

---

## Questions?

- 💬 **Discussions**: Use GitHub Discussions for questions
- 🐛 **Bugs**: Create an issue
- 📧 **Email**: For security concerns only

---

Thank you for contributing! 🙏

Every contribution, no matter how small, helps make this project better.
