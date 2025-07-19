// VLM Attention Visualizer - Client-side JavaScript

class AttentionVisualizer {
    constructor() {
        this.data = null;
        this.currentWord = null;
        this.currentWordIndex = null;
        this.currentLayer = 27;
        this.currentHead = 2;
        
        this.initializeElements();
        this.bindEvents();
        this.initialize();
    }

    initializeElements() {
        // Main containers
        this.loading = document.getElementById('loading');
        this.errorAlert = document.getElementById('error-alert');
        this.errorMessage = document.getElementById('error-message');
        this.mainContent = document.getElementById('main-content');

        // Controls
        this.tokenDisplay = document.getElementById('token-display');
        this.layerRange = document.getElementById('layer-range');
        this.headRange = document.getElementById('head-range');
        this.generateBtn = document.getElementById('generate-btn');
        this.clearBtn = document.getElementById('clear-btn');

        // Display elements
        this.layerValue = document.getElementById('layer-value');
        this.headValue = document.getElementById('head-value');
        this.currentWordSpan = document.getElementById('current-word');
        this.currentLayerSpan = document.getElementById('current-layer');
        this.currentHeadSpan = document.getElementById('current-head');

        // Images and visualization
        this.originalImage = document.getElementById('original-image');
        this.attentionOverlay = document.getElementById('attention-overlay');
        this.imageContainer = document.getElementById('image-container');
        this.attentionStatus = document.getElementById('attention-status');
        this.generationStatus = document.getElementById('generation-status');
    }

    bindEvents() {
        // Layer range
        this.layerRange.addEventListener('input', (e) => {
            this.currentLayer = parseInt(e.target.value);
            this.layerValue.textContent = this.currentLayer;
            this.updateCurrentSelection();
        });

        // Head range
        this.headRange.addEventListener('input', (e) => {
            this.currentHead = parseInt(e.target.value);
            this.headValue.textContent = this.currentHead;
            this.updateCurrentSelection();
        });

        // Generate button
        this.generateBtn.addEventListener('click', () => {
            this.generateAttentionMap();
        });

        // Clear button
        this.clearBtn.addEventListener('click', () => {
            this.clearAttention();
        });

        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !this.generateBtn.disabled) {
                this.generateAttentionMap();
            } else if (e.key === 'Escape' && !this.clearBtn.classList.contains('d-none')) {
                this.clearAttention();
            }
        });

        // Window resize handler to maintain overlay positioning
        window.addEventListener('resize', () => {
            if (!this.attentionOverlay.classList.contains('d-none')) {
                this.updateOverlaySize();
            }
        });
    }

    async initialize() {
        try {
            this.showLoading();
            
            const response = await fetch('/api/initialize');
            const result = await response.json();

            if (result.success) {
                this.data = result;
                this.setupControls();
                this.loadOriginalImage();
                this.hideLoading();
                this.showMainContent();
            } else {
                this.showError(result.error || 'Failed to initialize application');
            }
        } catch (error) {
            this.showError(`Network error: ${error.message}`);
        }
    }

    setupControls() {
        // Display tokens interactively
        this.displayTokens();

        // Set range limits
        this.layerRange.max = this.data.num_layers - 1;
        this.headRange.max = this.data.num_heads - 1;

        // Set default values if they exceed limits
        if (this.currentLayer >= this.data.num_layers) {
            this.currentLayer = this.data.num_layers - 1;
            this.layerRange.value = this.currentLayer;
            this.layerValue.textContent = this.currentLayer;
        }

        if (this.currentHead >= this.data.num_heads) {
            this.currentHead = this.data.num_heads - 1;
            this.headRange.value = this.currentHead;
            this.headValue.textContent = this.currentHead;
        }

        this.updateCurrentSelection();
    }

    displayTokens() {
        // Create prompt and token display
        let html = '';
        
        // Add prompt
        html += `<div class="prompt-text">Prompt: ${this.escapeHtml(this.data.prompt)}</div>`;
        html += '<div class="response-label">SmolVLM Output:</div>';
        html += '<div class="model-output">';
        
        // Add tokens preserving exact whitespace
        this.data.tokens.forEach((tokenData, index) => {
            if (tokenData.text === '\n') {
                // Render newlines as actual line breaks
                html += '<br>';
            } else if (tokenData.is_selectable) {
                // Render selectable tokens as clickable spans
                const escapedText = this.escapeHtml(tokenData.text);
                const escapedToken = this.escapeHtml(tokenData.text);
                
                html += `<span class="token" data-token='${escapedToken}' data-index="${tokenData.index}">${escapedText}</span>`;
            } else {
                // Render non-selectable tokens as plain text
                const escapedText = this.escapeHtml(tokenData.text);
                html += escapedText;
            }
        });
        
        html += '</div>';
        
        this.tokenDisplay.innerHTML = html;
        
        // Add click handlers to tokens
        this.tokenDisplay.querySelectorAll('.token').forEach(tokenElement => {
            tokenElement.addEventListener('click', (e) => {
                this.selectToken(e.target);
            });
        });
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    selectToken(tokenElement) {
        // Remove previous selection
        this.tokenDisplay.querySelectorAll('.token.selected').forEach(el => {
            el.classList.remove('selected');
        });
        
        // Select new token
        tokenElement.classList.add('selected');
        
        // Get the exact token text (unescaped) from the data attribute
        this.currentWord = this.unescapeHtml(tokenElement.dataset.token);
        this.currentWordIndex = parseInt(tokenElement.dataset.index);
        
        this.updateCurrentSelection();
        this.toggleGenerateButton();
    }

    unescapeHtml(text) {
        const div = document.createElement('div');
        div.innerHTML = text;
        return div.textContent || div.innerText || '';
    }

    loadOriginalImage() {
        this.originalImage.src = this.data.image_url;
        this.originalImage.onload = () => {
            this.imageContainer.classList.add('fade-in');
            
            // If there's an overlay visible, update its size after image loads
            if (!this.attentionOverlay.classList.contains('d-none')) {
                setTimeout(() => this.updateOverlaySize(), 10);
            }
        };
    }

    updateCurrentSelection() {
        this.currentWordSpan.textContent = this.currentWord || '-';
        this.currentLayerSpan.textContent = this.currentLayer;
        this.currentHeadSpan.textContent = this.currentHead;
    }

    toggleGenerateButton() {
        this.generateBtn.disabled = !this.currentWord;
    }

    async generateAttentionMap() {
        if (!this.currentWord) {
            this.showError('Please select a word first');
            return;
        }

        try {
            this.showGenerationStatus();
            this.generateBtn.disabled = true;

            const response = await fetch('/api/generate_attention_map', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    word: this.currentWord,
                    word_index: this.currentWordIndex,
                    layer: this.currentLayer,
                    head: this.currentHead
                })
            });

            const result = await response.json();

            if (result.success) {
                this.displayAttentionOverlay(result.heatmap, result.heatmap_width, result.heatmap_height);
            } else {
                this.showError(result.error || 'Failed to generate attention map');
            }
        } catch (error) {
            this.showError(`Network error: ${error.message}`);
        } finally {
            this.hideGenerationStatus();
            this.generateBtn.disabled = false;
        }
    }

    displayAttentionOverlay(heatmapBase64, width, height) {
        this.attentionOverlay.src = `data:image/png;base64,${heatmapBase64}`;
        this.attentionOverlay.onload = () => {
            // Wait for the image to fully render, then set exact dimensions
            setTimeout(() => {
                this.updateOverlaySize();
                
                // Show the overlay and status
                this.attentionOverlay.classList.remove('d-none');
                this.attentionOverlay.classList.add('fade-in');
                this.attentionStatus.classList.remove('d-none');
                this.clearBtn.classList.remove('d-none');
            }, 10); // Small delay to ensure rendering is complete
        };
    }

    clearAttention() {
        this.attentionOverlay.classList.add('d-none');
        this.attentionStatus.classList.add('d-none');
        this.clearBtn.classList.add('d-none');
    }

    updateOverlaySize() {
        // Wait for image to be fully rendered
        if (this.originalImage.naturalWidth === 0 || this.originalImage.naturalHeight === 0) {
            // Image not loaded yet, try again in a moment
            setTimeout(() => this.updateOverlaySize(), 50);
            return;
        }
        
        // Get the container and image positions
        const containerRect = this.imageContainer.getBoundingClientRect();
        const imageRect = this.originalImage.getBoundingClientRect();
        const computedStyle = getComputedStyle(this.originalImage);
        
        // Calculate the offset of the image relative to the container
        const offsetTop = imageRect.top - containerRect.top;
        const offsetLeft = imageRect.left - containerRect.left;
        
        // Set exact dimensions and position
        this.attentionOverlay.style.width = imageRect.width + 'px';
        this.attentionOverlay.style.height = imageRect.height + 'px';
        this.attentionOverlay.style.top = offsetTop + 'px';
        this.attentionOverlay.style.left = offsetLeft + 'px';
        
        // Match border radius exactly
        this.attentionOverlay.style.borderRadius = computedStyle.borderRadius;
        
        // Ensure object-fit properties match if any
        this.attentionOverlay.style.objectFit = computedStyle.objectFit || 'fill';
    }

    showLoading() {
        this.loading.classList.remove('d-none');
        this.hideError();
        this.hideMainContent();
    }

    hideLoading() {
        this.loading.classList.add('d-none');
    }

    showError(message) {
        this.errorMessage.textContent = message;
        this.errorAlert.classList.remove('d-none');
        this.hideLoading();
        this.hideMainContent();
    }

    hideError() {
        this.errorAlert.classList.add('d-none');
    }

    showMainContent() {
        this.mainContent.classList.remove('d-none');
        this.mainContent.classList.add('fade-in');
    }

    hideMainContent() {
        this.mainContent.classList.add('d-none');
    }

    showGenerationStatus() {
        this.generationStatus.classList.remove('d-none');
    }

    hideGenerationStatus() {
        this.generationStatus.classList.add('d-none');
    }


}

// Utility functions
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    // Add some visual feedback for interactions
    document.querySelectorAll('.btn').forEach(btn => {
        btn.addEventListener('click', function(e) {
            // Create ripple effect
            const ripple = document.createElement('span');
            const rect = this.getBoundingClientRect();
            const size = Math.max(rect.height, rect.width);
            ripple.style.width = ripple.style.height = size + 'px';
            ripple.style.left = (e.clientX - rect.left - size / 2) + 'px';
            ripple.style.top = (e.clientY - rect.top - size / 2) + 'px';
            ripple.classList.add('ripple');
            this.appendChild(ripple);
            
            setTimeout(() => {
                ripple.remove();
            }, 600);
        });
    });

    // Initialize the main application
    new AttentionVisualizer();
});

// Add ripple effect CSS
const style = document.createElement('style');
style.textContent = `
    .btn {
        position: relative;
        overflow: hidden;
    }
    
    .ripple {
        position: absolute;
        border-radius: 50%;
        background: rgba(255, 255, 255, 0.6);
        transform: scale(0);
        animation: ripple-animation 0.6s linear;
        pointer-events: none;
    }
    
    @keyframes ripple-animation {
        to {
            transform: scale(4);
            opacity: 0;
        }
    }
`;
document.head.appendChild(style); 