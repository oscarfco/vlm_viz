# VLM Attention Visualizer

A modern, interactive web application for visualizing Vision Language Model (VLM) attention patterns. This tool allows you to explore how VLMs attend to different regions of an image based on specific tokens/words in the output.

## Features

- 🎯 **Interactive Token Selection**: Click directly on words in the full model output (prompt + response) to select tokens
- 🧠 **Layer & Head Control**: Explore attention across different transformer layers and attention heads
- 🖼️ **Real-time Visualization**: Interactive attention heatmap overlays directly on the original image
- 🎨 **Modern UI**: Clean, responsive interface with smooth animations and intuitive controls
- ⚡ **Fast Processing**: Efficient backend processing with real-time feedback

## Prerequisites

Before running the application, make sure you have the following files in the project directory:

- `attention_array.npy`: NumPy array containing attention weights from your VLM
- `decoded_tokens.npy`: NumPy array containing the decoded tokens from your VLM
- `image.png`: The image file you want to analyze (should be in the project root directory)

## Installation

1. **Clone or download the project files**

2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Ensure your data files are in place**:
   - Place `attention_array.npy` in the project root directory
   - Place `decoded_tokens.npy` in the project root directory
   - Place `image.png` in the project root directory

## Usage

1. **Start the Flask application**:
   ```bash
   python app.py
   ```

2. **Open your web browser** and navigate to:
   ```
   http://localhost:5001
   ```

3. **Use the interface**:
   - Wait for the application to load your attention data
   - View the extracted prompt and full assistant response above the image
   - Click on any word/token in the assistant response to select it
   - Selected tokens preserve exact whitespace and formatting
   - Adjust the layer and attention head using the sliders in the left panel
   - Click "Generate Attention Map" to overlay the attention heatmap
   - Use "Clear Attention" button or press Escape to remove the overlay

## How It Works

### Data Processing
The application expects your attention data in a specific format:
- **Attention Array**: Should contain attention weights with shape `[layers, heads, sequence_length, sequence_length]`
- **Decoded Tokens**: List of tokens corresponding to the model's output

### Visualization Process
1. The app loads your attention data and extracts available tokens
2. When you select a word, it finds the corresponding token index
3. It extracts attention weights for that token across all image patches
4. The attention weights are reshaped and mapped to image coordinates
5. A transparent heatmap is generated and overlaid directly on the original image in the browser

### Image Processing
- The app uses a local `image.png` file for visualization
- Images are processed in patches (512x512 pixels by default)
- Attention weights are interpolated to create smooth overlays

## Customization

### Changing the Image
To use a different image, simply replace the `image.png` file in the project root directory with your desired image (keeping the same filename), or modify the image paths in the `initialize()` function in `app.py`.

### Adjusting Parameters
Key parameters that can be modified in `app.py`:

- `smolvlm_patch_size`: Size of image patches (default: 512)
- `smolvlm_token_patch_size`: Token patch size (default: 8)
- `num_rows`, `num_cols`: Grid dimensions for patches (default: 3x4)

### UI Customization
- Modify `static/css/style.css` for visual styling
- Update `templates/index.html` for layout changes
- Adjust `static/js/app.js` for interaction behavior

## API Endpoints

The application provides the following API endpoints:

### GET `/api/initialize`
Initializes the application and returns available tokens and model dimensions.

**Response:**
```json
{
  "success": true,
  "tokens": ["list", "of", "available", "tokens"],
  "num_layers": 32,
  "num_heads": 16,
  "image_url": "https://example.com/image.jpg"
}
```

### POST `/api/generate_attention_map`
Generates an attention map for the specified parameters.

**Request:**
```json
{
  "word": "selected_word",
  "layer": 27,
  "head": 2
}
```

**Response:**
```json
{
  "success": true,
  "image": "base64_encoded_image_data",
  "word_index": 42,
  "layer": 27,
  "head": 2
}
```

## Technical Stack

- **Backend**: Flask (Python)
- **Frontend**: HTML5, CSS3, JavaScript (ES6+)
- **Styling**: Bootstrap 5 + Custom CSS
- **Data Processing**: NumPy, Matplotlib, PIL
- **Image Processing**: PIL (Python Imaging Library)

## File Structure

```
vlm_viz/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── test_data.py          # Data format validation script
├── attention_array.npy   # Your attention data (not included)
├── decoded_tokens.npy    # Your token data (not included)
├── image.png             # Your image file (not included)
├── templates/
│   └── index.html        # Main HTML template
└── static/
    ├── css/
    │   └── style.css     # Custom styling
    └── js/
        └── app.js        # Client-side JavaScript
```

## Troubleshooting

### Common Issues

1. **"Failed to load data" error**:
   - Ensure `attention_array.npy` and `decoded_tokens.npy` are in the project root
   - Check that the files are valid NumPy arrays

2. **"Word not found in tokens" error**:
   - The selected word might not exist in your decoded tokens
   - Check the token list for exact matches (including spaces and punctuation)

3. **Image loading issues**:
   - Ensure `image.png` exists in the project root directory
   - Check that the image file is a valid PNG, JPEG, or other PIL-supported format
   - The app will automatically copy the image to the `static/` directory for web serving

4. **Port already in use**:
   - Change the port in `app.py`: `app.run(debug=True, port=5002)` (or any other available port)

### Performance Tips

- For large attention arrays, consider reducing the image resolution
- Use fewer layers/heads for faster processing
- Optimize patch sizes based on your specific model architecture

## Contributing

Feel free to submit issues, feature requests, or pull requests to improve the application.

## License

This project is open source and available under the MIT License. 