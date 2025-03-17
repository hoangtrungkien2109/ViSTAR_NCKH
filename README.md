# ViSTAR (Vietnamese Sign Language Translation and Recognition)
<div align="center">

[한국어]() | [日本語]() | [Русский]() | [Deutsch]() | [Français]() | [Español]() | [Português]() | [Türkçe]() | [Tiếng Việt]() | [العربية]()


<div>
    <img src="https://zenodo.org/badge/264818686.svg" alt="YOLOv5 Citation">
    <img src="https://img.shields.io/docker/pulls/ultralytics/yolov5?logo=docker" alt="Docker Pulls">
    <img alt="Discord" src="https://img.shields.io/discord/1089800235347353640?logo=discord&logoColor=white&label=Discord&color=blue">
    <img alt="Ultralytics Forums" src="https://img.shields.io/discourse/users?server=https%3A%2F%2Fcommunity.ultralytics.com&logo=discourse&label=Forums&color=blue">
    <img alt="Ultralytics Reddit" src="https://img.shields.io/reddit/subreddit-subscribers/ultralytics?style=flat&logo=reddit&logoColor=white&label=Reddit&color=blue">
    <br>
    <img src="https://assets.paperspace.io/img/gradient-badge.svg" alt="Run on Gradient">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab">
    <img src="https://kaggle.com/static/images/open-in-kaggle.svg" alt="Open In Kaggle">
</div>

  <br>
</div>

![ViSTAR Logo](src/VISTAR.webp)
A comprehensive system for real-time sign language translation using AI and computer vision. ViSTAR enables bidirectional translation between text and sign language through video frames.

## 🌟 Features

- **Text to Sign Language**: Convert written text into sign language video frames
- **Sign Language to Text**: Translate sign language gestures into written text
- **Real-time Processing**: WebSocket-based streaming for smooth real-time translations
- **User Management**: Secure authentication and user data management
- **Modern UI**: Responsive interface built with TailwindCSS

## 🏗️ Architecture

The project is structured into several key components:

- **Frontend (src/fe/)**: FastAPI-based web interface with WebSocket support
- **AI Service (src/ai/)**: Machine learning models for sign language processing
- **Backend (src/be/)**: Core business logic and data management
- **Streaming (src/streaming/)**: gRPC-based streaming service for real-time communication

## 🔧 Technologies

- **Backend Framework**: FastAPI, Django
- **AI/ML**: PyTorch, MediaPipe
- **Communication**: gRPC, WebSockets
- **Frontend**: TailwindCSS
- **Database**: SQLAlchemy
- **Search**: Elasticsearch
- **Authentication**: Passlib, bcrypt

## 📋 Prerequisites

- Python 3.x
- Node.js (for TailwindCSS)
- Docker (optional, for containerized deployment)

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ViSTAR.git
cd ViSTAR
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Set up the frontend assets:
```bash
cd src/fe
npm install
```

## 💻 Development Setup

1. Start the streaming service:
```bash
python src/streaming/main.py
```

2. Launch the AI service:
```bash
python src/ai/main.py
```

3. Run the frontend server:
```bash
cd src/fe
python main.py
```

## 🐳 Docker Deployment

The project includes Docker support for both development and production environments:

Development:
```bash
docker-compose -f docker-compose.dev.yaml up
```

Production:
```bash
docker-compose -f docker-compose.prod.yaml up
```

## 🔐 Environment Variables

Create a `.env` file in the root directory with the following variables:
```
DATABASE_URL=your_database_url
ELASTICSEARCH_URL=your_elasticsearch_url
SECRET_KEY=your_secret_key
```

## 📁 Project Structure

```
ViSTAR/
├── src/
│   ├── ai/                 # AI/ML models and services
│   ├── be/                 # Backend services
│   ├── fe/                 # Frontend application
│   ├── streaming/          # gRPC streaming services
│   └── init_data/         # Initial data and setup scripts
├── data/                  # Data storage
├── docker-compose.*.yaml  # Docker configurations
└── requirements.txt       # Python dependencies
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
