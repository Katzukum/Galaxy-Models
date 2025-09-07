// Training Tab JavaScript
class Training {
    constructor() {
        this.currentTrainingId = null;
        this.statusUpdateInterval = null;
        this.autoScroll = true;
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.loadAvailableCSVFiles();
        this.loadTrainingHistory();
    }

    setupEventListeners() {
        // Model type selection
        const modelTypeSelect = document.getElementById('model-type');
        if (modelTypeSelect) {
            modelTypeSelect.addEventListener('change', (e) => {
                this.updateModelHelp(e.target.value);
            });
        }

        // File input handling
        const fileInput = document.getElementById('csv-file');
        if (fileInput) {
            fileInput.addEventListener('change', (e) => {
                this.handleFileSelection(e);
            });

            // Drag and drop functionality
            const fileDisplay = document.querySelector('.file-input-display');
            if (fileDisplay) {
                fileDisplay.addEventListener('dragover', (e) => {
                    e.preventDefault();
                    fileDisplay.style.borderColor = 'var(--accent-color)';
                });

                fileDisplay.addEventListener('dragleave', (e) => {
                    e.preventDefault();
                    fileDisplay.style.borderColor = 'var(--border-color)';
                });

                fileDisplay.addEventListener('drop', (e) => {
                    e.preventDefault();
                    fileDisplay.style.borderColor = 'var(--border-color)';
                    
                    const files = e.dataTransfer.files;
                    if (files.length > 0 && files[0].type === 'text/csv') {
                        fileInput.files = files;
                        this.handleFileSelection({ target: fileInput });
                    }
                });
            }
        }

        // Form submission
        const trainingForm = document.getElementById('training-form');
        if (trainingForm) {
            trainingForm.addEventListener('submit', (e) => {
                e.preventDefault();
                this.startTraining();
            });
        }

        // Training control buttons
        const stopTrainingBtn = document.getElementById('stop-training-btn');
        if (stopTrainingBtn) {
            stopTrainingBtn.addEventListener('click', () => {
                this.stopTraining();
            });
        }

        const viewLogsBtn = document.getElementById('view-logs-btn');
        if (viewLogsBtn) {
            viewLogsBtn.addEventListener('click', () => {
                this.toggleLogsSection();
            });
        }

        // Logs controls
        const clearLogsBtn = document.getElementById('clear-logs-btn');
        if (clearLogsBtn) {
            clearLogsBtn.addEventListener('click', () => {
                this.clearLogs();
            });
        }

        const exportLogsBtn = document.getElementById('export-logs-btn');
        if (exportLogsBtn) {
            exportLogsBtn.addEventListener('click', () => {
                this.exportLogs();
            });
        }

        const toggleAutoScrollBtn = document.getElementById('toggle-auto-scroll');
        if (toggleAutoScrollBtn) {
            toggleAutoScrollBtn.addEventListener('click', () => {
                this.toggleAutoScroll();
            });
        }

        // Reset parameters button
        const resetParamsBtn = document.getElementById('reset-params-btn');
        if (resetParamsBtn) {
            resetParamsBtn.addEventListener('click', () => {
                this.resetParametersToDefaults();
            });
        }
    }

    updateModelHelp(modelType) {
        const helpTexts = document.querySelectorAll('.help-text');
        helpTexts.forEach(text => {
            text.style.display = 'none';
        });

        const activeHelp = document.querySelector(`[data-model="${modelType}"]`);
        if (activeHelp) {
            activeHelp.style.display = 'block';
        }

        // Show/hide parameter sections based on model type
        this.updateParameterSections(modelType);
    }

    updateParameterSections(modelType) {
        // Hide all parameter sections
        const allParams = document.querySelectorAll('.model-params');
        allParams.forEach(params => {
            params.style.display = 'none';
        });

        // Show the relevant parameter section
        const targetParams = document.getElementById(`${modelType}-params`);
        if (targetParams) {
            targetParams.style.display = 'block';
        }
    }

    handleFileSelection(event) {
        const file = event.target.files[0];
        const fileDisplay = document.querySelector('.file-input-display');
        const fileText = document.querySelector('.file-input-text');
        const fileIcon = document.querySelector('.file-input-icon');

        if (file) {
            fileDisplay.classList.add('has-file');
            fileText.textContent = file.name;
            fileIcon.textContent = '✅';
        } else {
            fileDisplay.classList.remove('has-file');
            fileText.textContent = 'Choose CSV file or drag & drop';
            fileIcon.textContent = '📁';
        }
    }

    async loadAvailableCSVFiles() {
        try {
            const csvFiles = await eel.get_available_csv_files()();
            this.displayCSVFiles(csvFiles);
        } catch (error) {
            console.error('Error loading CSV files:', error);
            this.displayCSVFilesError();
        }
    }

    displayCSVFiles(csvFiles) {
        const container = document.getElementById('available-csv-files');
        if (!container) return;

        if (csvFiles.length === 0) {
            container.innerHTML = '<div class="no-files">No CSV files found in the project directory.</div>';
            return;
        }

        container.innerHTML = csvFiles.map(file => `
            <div class="csv-file-item" data-path="${file.path}">
                <div class="csv-file-info">
                    <div class="csv-file-name">${file.name}</div>
                    <div class="csv-file-details">
                        Modified: ${new Date(file.modified).toLocaleString()}
                    </div>
                </div>
                <div class="csv-file-size">${this.formatFileSize(file.size)}</div>
            </div>
        `).join('');

        // Add click handlers for CSV file selection
        container.querySelectorAll('.csv-file-item').forEach(item => {
            item.addEventListener('click', () => {
                this.selectCSVFile(item);
            });
        });
    }

    displayCSVFilesError() {
        const container = document.getElementById('available-csv-files');
        if (container) {
            container.innerHTML = '<div class="error">Error loading CSV files. Please check the console for details.</div>';
        }
    }

    selectCSVFile(item) {
        // Remove previous selection
        document.querySelectorAll('.csv-file-item').forEach(i => {
            i.classList.remove('selected');
        });

        // Add selection to clicked item
        item.classList.add('selected');

        // Update file input
        const filePath = item.dataset.path;
        const fileInput = document.getElementById('csv-file');
        if (fileInput) {
            // Create a fake file object for the selected CSV
            const fileName = filePath.split('/').pop();
            const fakeFile = new File([''], fileName, { type: 'text/csv' });
            const dataTransfer = new DataTransfer();
            dataTransfer.items.add(fakeFile);
            fileInput.files = dataTransfer.files;
            
            // Update display
            this.handleFileSelection({ target: fileInput });
        }
    }

    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    fileToBase64(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.readAsDataURL(file);
            reader.onload = () => {
                // Remove the data URL prefix (data:type;base64,)
                const base64 = reader.result.split(',')[1];
                resolve(base64);
            };
            reader.onerror = error => reject(error);
        });
    }

    getTrainingParameters(modelType) {
        const params = {};

        if (modelType === 'transformer') {
            // Validate transformer parameters
            const dModel = parseInt(document.getElementById('transformer-d-model').value);
            const nhead = parseInt(document.getElementById('transformer-nhead').value);
            const layers = parseInt(document.getElementById('transformer-layers').value);
            
            if (dModel % nhead !== 0) {
                throw new Error(`d_model (${dModel}) must be divisible by nhead (${nhead})`);
            }
            params.model_params = {
                d_model: parseInt(document.getElementById('transformer-d-model').value),
                nhead: parseInt(document.getElementById('transformer-nhead').value),
                num_encoder_layers: parseInt(document.getElementById('transformer-layers').value),
                dim_feedforward: parseInt(document.getElementById('transformer-feedforward').value),
                dropout: parseFloat(document.getElementById('transformer-dropout').value)
            };
            params.data_params = {
                sequence_length: parseInt(document.getElementById('transformer-sequence-length').value)
            };
            params.train_params = {
                learning_rate: parseFloat(document.getElementById('transformer-learning-rate').value),
                epochs: parseInt(document.getElementById('transformer-epochs').value),
                batch_size: parseInt(document.getElementById('transformer-batch-size').value)
            };
        } else if (modelType === 'nn') {
            const hiddenLayers = parseInt(document.getElementById('nn-hidden-layers').value);
            const architecture = [];
            
            // Build architecture dynamically based on number of layers
            let inputFeatures = 'auto'; // Will be set based on data
            for (let i = 0; i < hiddenLayers; i++) {
                const neurons = i === 0 ? 
                    parseInt(document.getElementById('nn-neurons-layer1').value) :
                    (i === 1 ? parseInt(document.getElementById('nn-neurons-layer2').value) : 32);
                
                architecture.push({
                    type: 'Linear',
                    in_features: inputFeatures,
                    out_features: neurons
                });
                architecture.push({ type: 'ReLU' });
                if (i < hiddenLayers - 1) {
                    architecture.push({
                        type: 'Dropout',
                        p: parseFloat(document.getElementById('nn-dropout').value)
                    });
                }
                inputFeatures = neurons;
            }
            
            // Add output layer
            architecture.push({
                type: 'Linear',
                in_features: inputFeatures,
                out_features: 1
            });

            params.architecture = architecture;
            params.train_params = {
                optimizer: document.getElementById('nn-optimizer').value,
                loss: document.getElementById('nn-loss').value,
                learning_rate: parseFloat(document.getElementById('nn-learning-rate').value),
                epochs: parseInt(document.getElementById('nn-epochs').value)
            };
        } else if (modelType === 'xgboost') {
            params.model_params = {
                objective: document.getElementById('xgb-objective').value,
                eval_metric: document.getElementById('xgb-eval-metric').value,
                n_estimators: parseInt(document.getElementById('xgb-n-estimators').value),
                learning_rate: parseFloat(document.getElementById('xgb-learning-rate').value),
                max_depth: parseInt(document.getElementById('xgb-max-depth').value),
                use_label_encoder: false
            };
            params.label_params = {
                look_ahead_periods: document.getElementById('xgb-look-ahead').value.split(',').map(x => parseInt(x.trim())),
                min_tick_change: parseInt(document.getElementById('xgb-min-tick').value)
            };
        }

        return params;
    }

    resetParametersToDefaults() {
        // Reset Transformer parameters
        document.getElementById('transformer-d-model').value = 64;
        document.getElementById('transformer-nhead').value = 4;
        document.getElementById('transformer-layers').value = 3;
        document.getElementById('transformer-feedforward').value = 128;
        document.getElementById('transformer-dropout').value = 0.1;
        document.getElementById('transformer-sequence-length').value = 60;
        document.getElementById('transformer-learning-rate').value = 0.0005;
        document.getElementById('transformer-epochs').value = 25;
        document.getElementById('transformer-batch-size').value = 32;

        // Reset Neural Network parameters
        document.getElementById('nn-hidden-layers').value = 2;
        document.getElementById('nn-neurons-layer1').value = 64;
        document.getElementById('nn-neurons-layer2').value = 32;
        document.getElementById('nn-dropout').value = 0.3;
        document.getElementById('nn-optimizer').value = 'Adam';
        document.getElementById('nn-loss').value = 'MSELoss';
        document.getElementById('nn-learning-rate').value = 0.001;
        document.getElementById('nn-epochs').value = 100;

        // Reset XGBoost parameters
        document.getElementById('xgb-objective').value = 'multi:softmax';
        document.getElementById('xgb-eval-metric').value = 'mlogloss';
        document.getElementById('xgb-n-estimators').value = 150;
        document.getElementById('xgb-learning-rate').value = 0.1;
        document.getElementById('xgb-max-depth').value = 4;
        document.getElementById('xgb-look-ahead').value = '3,5';
        document.getElementById('xgb-min-tick').value = 20;

        this.showSuccess('Parameters reset to default values');
    }

    async startTraining() {
        const modelTypeElement = document.getElementById('model-type');
        const modelNameElement = document.getElementById('model-name');
        const fileInput = document.getElementById('csv-file');
        const startBtn = document.getElementById('start-training-btn');

        // Get values with null checks
        const modelType = modelTypeElement ? modelTypeElement.value : '';
        const modelName = modelNameElement ? modelNameElement.value.trim() : '';

        // Validation
        if (!modelType) {
            this.showError('Please select a model type.');
            return;
        }

        // Check for data file - either from available files list or file input
        let csvPath = '';
        
        // First, check if a file is selected from the available files list
        const selectedFileItem = document.querySelector('.csv-file-item.selected');
        if (selectedFileItem) {
            csvPath = selectedFileItem.dataset.path;
        } 
        // If no file selected from list, check if a file was uploaded via file input
        else if (fileInput.files && fileInput.files.length > 0 && fileInput.files[0].size > 0) {
            // For uploaded files, we need to save them to the server first
            try {
                // Convert file to base64 and send to server
                const base64 = await this.fileToBase64(fileInput.files[0]);
                const uploadResult = await eel.upload_csv_file({
                    'name': fileInput.files[0].name,
                    'content': base64
                })();
                
                if (uploadResult.success) {
                    csvPath = uploadResult.file_path;
                } else {
                    this.showError(`File upload failed: ${uploadResult.error}`);
                    return;
                }
            } catch (error) {
                console.error('File upload error:', error);
                this.showError('File upload failed');
                return;
            }
        } 
        // If neither, show error
        else {
            this.showError('Please select a CSV file from the available files list or upload a new file.');
            return;
        }

        try {
            // Update UI
            startBtn.disabled = true;
            startBtn.classList.add('loading');
            this.hideError();

            // Get training parameters
            let trainingParams;
            try {
                trainingParams = this.getTrainingParameters(modelType);
            } catch (error) {
                this.showError(`Parameter validation error: ${error.message}`);
                return;
            }

            // Start training with optional model name and parameters
            const result = await eel.start_training(modelType, csvPath, null, modelName || null, trainingParams)();
            
            if (result.error) {
                this.showError(result.error);
                return;
            }

            this.currentTrainingId = result.training_id;
            this.showTrainingProgress();
            this.startStatusUpdates();

        } catch (error) {
            console.error('Error starting training:', error);
            this.showError('Failed to start training. Please check the console for details.');
        } finally {
            startBtn.disabled = false;
            startBtn.classList.remove('loading');
        }
    }

    showTrainingProgress() {
        const progressSection = document.getElementById('training-progress-section');
        if (progressSection) {
            progressSection.style.display = 'block';
            progressSection.scrollIntoView({ behavior: 'smooth' });
        }
    }

    startStatusUpdates() {
        if (this.statusUpdateInterval) {
            clearInterval(this.statusUpdateInterval);
        }

        this.statusUpdateInterval = setInterval(async () => {
            if (this.currentTrainingId) {
                await this.updateTrainingStatus();
            }
        }, 2000); // Update every 2 seconds
    }

    async updateTrainingStatus() {
        try {
            const status = await eel.get_training_status(this.currentTrainingId)();
            
            if (status.error) {
                console.error('Error getting training status:', status.error);
                return;
            }

            this.updateProgressUI(status);

            // Check if training is complete
            if (['completed', 'failed', 'error', 'stopped'].includes(status.status)) {
                this.stopStatusUpdates();
                this.loadTrainingHistory(); // Refresh history
            }

        } catch (error) {
            console.error('Error updating training status:', error);
        }
    }

    updateProgressUI(status) {
        // Update status badge
        const statusBadge = document.getElementById('training-status');
        if (statusBadge) {
            statusBadge.textContent = status.status;
            statusBadge.className = `status-badge ${status.status}`;
        }

        // Update model type badge
        const modelTypeBadge = document.getElementById('training-model-type');
        if (modelTypeBadge) {
            const displayText = status.model_name ? `${status.model_type} (${status.model_name})` : status.model_type;
            modelTypeBadge.textContent = displayText;
        }

        // Update progress bar
        const progressFill = document.getElementById('progress-fill');
        const progressText = document.getElementById('progress-text');
        if (progressFill && progressText) {
            progressFill.style.width = `${status.progress}%`;
            progressText.textContent = `${Math.round(status.progress)}%`;
        }

        // Update progress details
        const progressMessage = document.getElementById('progress-message');
        if (progressMessage) {
            progressMessage.textContent = status.message;
        }

        const progressStartTime = document.getElementById('progress-start-time');
        if (progressStartTime && status.start_time) {
            progressStartTime.textContent = new Date(status.start_time).toLocaleString();
        }

        const progressDuration = document.getElementById('progress-duration');
        if (progressDuration && status.start_time) {
            const startTime = new Date(status.start_time);
            const currentTime = new Date();
            const duration = currentTime - startTime;
            progressDuration.textContent = this.formatDuration(duration);
        }

        // Show/hide stop button
        const stopBtn = document.getElementById('stop-training-btn');
        if (stopBtn) {
            stopBtn.style.display = status.status === 'running' ? 'block' : 'none';
        }

        // Update logs if training is running
        if (status.status === 'running') {
            this.updateLogs();
        }
    }

    formatDuration(milliseconds) {
        const seconds = Math.floor(milliseconds / 1000);
        const minutes = Math.floor(seconds / 60);
        const hours = Math.floor(minutes / 60);

        if (hours > 0) {
            return `${hours}h ${minutes % 60}m ${seconds % 60}s`;
        } else if (minutes > 0) {
            return `${minutes}m ${seconds % 60}s`;
        } else {
            return `${seconds}s`;
        }
    }

    async updateLogs() {
        if (!this.currentTrainingId) return;

        try {
            const logs = await eel.get_training_logs(this.currentTrainingId, 50)();
            
            if (logs.error) {
                console.error('Error getting training logs:', logs.error);
                return;
            }

            this.displayLogs(logs);

        } catch (error) {
            console.error('Error updating logs:', error);
        }
    }

    displayLogs(logs) {
        const logsContainer = document.getElementById('training-logs');
        if (!logsContainer) return;

        logsContainer.innerHTML = logs.map(log => `
            <div class="log-entry ${this.getLogEntryClass(log.message)}">
                <span class="log-timestamp">[${new Date(log.timestamp).toLocaleTimeString()}]</span>
                <span class="log-message">${this.escapeHtml(log.message)}</span>
            </div>
        `).join('');

        // Auto-scroll to bottom if enabled
        if (this.autoScroll) {
            logsContainer.scrollTop = logsContainer.scrollHeight;
        }
    }

    getLogEntryClass(message) {
        const lowerMessage = message.toLowerCase();
        if (lowerMessage.includes('error') || lowerMessage.includes('failed')) {
            return 'error';
        } else if (lowerMessage.includes('warning') || lowerMessage.includes('warn')) {
            return 'warning';
        } else if (lowerMessage.includes('success') || lowerMessage.includes('completed')) {
            return 'success';
        }
        return '';
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    async stopTraining() {
        if (!this.currentTrainingId) return;

        try {
            const result = await eel.stop_training(this.currentTrainingId)();
            
            if (result.error) {
                this.showError(result.error);
            } else {
                this.showSuccess('Training stopped successfully.');
            }

        } catch (error) {
            console.error('Error stopping training:', error);
            this.showError('Failed to stop training.');
        }
    }

    stopStatusUpdates() {
        if (this.statusUpdateInterval) {
            clearInterval(this.statusUpdateInterval);
            this.statusUpdateInterval = null;
        }
    }

    toggleLogsSection() {
        const logsSection = document.getElementById('training-logs-section');
        if (logsSection) {
            const isVisible = logsSection.style.display !== 'none';
            logsSection.style.display = isVisible ? 'none' : 'block';
            
            if (!isVisible) {
                logsSection.scrollIntoView({ behavior: 'smooth' });
                this.updateLogs();
            }
        }
    }

    clearLogs() {
        const logsContainer = document.getElementById('training-logs');
        if (logsContainer) {
            logsContainer.innerHTML = '<div class="log-entry"><span class="log-timestamp">[00:00:00]</span><span class="log-message">Logs cleared...</span></div>';
        }
    }

    exportLogs() {
        const logsContainer = document.getElementById('training-logs');
        if (!logsContainer) return;

        const logs = Array.from(logsContainer.querySelectorAll('.log-entry')).map(entry => {
            const timestamp = entry.querySelector('.log-timestamp').textContent;
            const message = entry.querySelector('.log-message').textContent;
            return `${timestamp} ${message}`;
        }).join('\n');

        const blob = new Blob([logs], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `training-logs-${new Date().toISOString().split('T')[0]}.txt`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }

    toggleAutoScroll() {
        this.autoScroll = !this.autoScroll;
        const toggleBtn = document.getElementById('toggle-auto-scroll');
        if (toggleBtn) {
            toggleBtn.classList.toggle('active', this.autoScroll);
        }
    }

    async loadTrainingHistory() {
        // This would load from a persistent storage or API
        // For now, we'll show a placeholder
        const historyContainer = document.getElementById('training-history');
        if (historyContainer) {
            historyContainer.innerHTML = `
                <div class="history-item">
                    <div class="history-header">
                        <div class="history-title">Training History</div>
                        <div class="history-status completed">Completed</div>
                    </div>
                    <div class="history-details">
                        <div class="history-detail">
                            <div class="history-detail-label">Model Type</div>
                            <div class="history-detail-value">Transformer</div>
                        </div>
                        <div class="history-detail">
                            <div class="history-detail-label">Duration</div>
                            <div class="history-detail-value">2m 34s</div>
                        </div>
                        <div class="history-detail">
                            <div class="history-detail-label">Status</div>
                            <div class="history-detail-value">Completed</div>
                        </div>
                    </div>
                </div>
            `;
        }
    }

    showError(message) {
        // Remove existing error messages
        this.hideError();
        
        // Create error message
        const errorDiv = document.createElement('div');
        errorDiv.className = 'error';
        errorDiv.textContent = message;
        
        // Insert after the form
        const form = document.getElementById('training-form');
        if (form) {
            form.parentNode.insertBefore(errorDiv, form.nextSibling);
        }
    }

    hideError() {
        const existingError = document.querySelector('.error');
        if (existingError) {
            existingError.remove();
        }
    }

    showSuccess(message) {
        // Create success message
        const successDiv = document.createElement('div');
        successDiv.className = 'success';
        successDiv.style.background = 'linear-gradient(45deg, #00ff00, #00cc00)';
        successDiv.style.color = '#000';
        successDiv.style.padding = '20px';
        successDiv.style.borderRadius = '15px';
        successDiv.style.margin = '30px 0';
        successDiv.style.textAlign = 'center';
        successDiv.style.border = '1px solid #00cc00';
        successDiv.style.boxShadow = '0 0 20px rgba(0, 255, 0, 0.3)';
        successDiv.style.backdropFilter = 'blur(10px)';
        successDiv.textContent = message;
        
        // Insert after the form
        const form = document.getElementById('training-form');
        if (form) {
            form.parentNode.insertBefore(successDiv, form.nextSibling);
            
            // Remove after 5 seconds
            setTimeout(() => {
                if (successDiv.parentNode) {
                    successDiv.parentNode.removeChild(successDiv);
                }
            }, 5000);
        }
    }

    static loadTraining() {
        console.log('Loading Training tab...');
        // Initialize training functionality when tab is loaded
        if (!window.trainingInstance) {
            window.trainingInstance = new Training();
        }
    }
}
