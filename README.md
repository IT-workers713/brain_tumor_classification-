# NeuroScan AI - Flask Version

Professional Flask interface for multi-class brain tumor MRI classification using an EfficientNet-based `effnet.h5` model.

## Features

- Flask upload workflow
- Professional medical-style UI
- Multi-class prediction output
- Ranked probabilities with progress bars
- Original image preview
- Preprocessed model-input preview
- Compatibility patch for `DepthwiseConv2D(groups=1)` issue

## Project Structure

```bash
neuroscan_flask/
├── app.py
├── effnet.h5                # place your model here
├── requirements.txt
├── README.md
├── uploads/
├── static/
│   └── css/
│       └── style.css
└── templates/
    ├── base.html
    ├── index.html
    └── result.html
```

## Installation

```bash
python -m venv venv
```

### Windows

```bash
venv\Scripts\activate
```

### macOS / Linux

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Add the model

Put your `effnet.h5` file in the project root, next to `app.py`.

## Run

```bash
python app.py
```

Then open:

```bash
http://127.0.0.1:5000
```

## Important note

This application is intended for research, educational, and demonstration use only. It is not a medical diagnosis tool.
