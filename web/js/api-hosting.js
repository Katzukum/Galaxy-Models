// API Hosting Tab JavaScript
class APIHosting {
    constructor() {
        this.currentAPI = null;
        this.apiProcess = null;
        this.statusUpdateInterval = null;
        this.logUpdateInterval = null;
        this.autoScroll = true;
        this.init();
    }

    init() {
        this.loadAPIHostingContent();
    }

    async loadAPIHostingContent() {
        try {
            const response = await fetch('tabs/api-hosting/api-hosting.html');
            const html = await response.text();
            document.getElementById('api-hosting-content').innerHTML = html;
            
            // Wait a bit for DOM to be ready
            setTimeout(() => {
                this.setupEventListeners();
                this.loadAvailableModels();
                this.loadAPIHistory();
            }, 100);
        } catch (error) {
            console.error('Error loading API hosting content:', error);
        }
    }

    setupEventListeners() {
        // Model selection
        const modelSelect = document.getElementById('api-model');
        if (modelSelect) {
            modelSelect.addEventListener('change', (e) => {
                this.handleModelSelection(e.target.value);
            });
        }

        // API control buttons
        const startBtn = document.getElementById('start-api-btn');
        if (startBtn) {
            startBtn.addEventListener('click', () => {
                this.startAPIServer();
            });
        }

        const stopBtn = document.getElementById('stop-api-btn');
        if (stopBtn) {
            stopBtn.addEventListener('click', () => {
                this.stopAPIServer();
            });
        }

        const testBtn = document.getElementById('test-api-btn');
        if (testBtn) {
            testBtn.addEventListener('click', () => {
                this.toggleTestingSection();
            });
        }

        // Testing controls
        const sendTestBtn = document.getElementById('send-test-btn');
        if (sendTestBtn) {
            sendTestBtn.addEventListener('click', () => {
                this.sendTestRequest();
            });
        }

        const clearTestBtn = document.getElementById('clear-test-btn');
        if (clearTestBtn) {
            clearTestBtn.addEventListener('click', () => {
                this.clearTestResults();
            });
        }

        // Logs controls
        const clearLogsBtn = document.getElementById('clear-api-logs-btn');
        if (clearLogsBtn) {
            clearLogsBtn.addEventListener('click', () => {
                this.clearLogs();
            });
        }

        const exportLogsBtn = document.getElementById('export-api-logs-btn');
        if (exportLogsBtn) {
            exportLogsBtn.addEventListener('click', () => {
                this.exportLogs();
            });
        }

        const toggleAutoScrollBtn = document.getElementById('toggle-api-auto-scroll');
        if (toggleAutoScrollBtn) {
            toggleAutoScrollBtn.addEventListener('click', () => {
                this.toggleAutoScroll();
            });
        }

        // Other controls
        const viewDocsBtn = document.getElementById('view-docs-btn');
        if (viewDocsBtn) {
            viewDocsBtn.addEventListener('click', () => {
                this.openAPIDocs();
            });
        }

        const copyUrlBtn = document.getElementById('copy-url-btn');
        if (copyUrlBtn) {
            copyUrlBtn.addEventListener('click', () => {
                this.copyAPIUrl();
            });
        }
    }

    async loadAvailableModels() {
        console.log('[DEBUG] loadAvailableModels() called');
        try {
            console.log('[DEBUG] Calling eel.get_available_models()');
            const models = await eel.get_available_models()();
            console.log('[DEBUG] get_available_models() returned:', models);
            
            const select = document.getElementById('api-model');
            console.log('[DEBUG] Model select element:', select);
            
            if (select) {
                select.innerHTML = '<option value="">Choose a model to host...</option>';
                models.forEach((model, index) => {
                    console.log(`[DEBUG] Processing model ${index}:`, model);
                    const option = document.createElement('option');
                    option.value = model.config_path;
                    option.textContent = `${model.name} (${model.type})`;
                    console.log(`[DEBUG] Created option with value: "${option.value}" and text: "${option.textContent}"`);
                    select.appendChild(option);
                });
                console.log('[DEBUG] All model options added to select');
            }
        } catch (error) {
            console.error('[DEBUG] Error in loadAvailableModels:', error);
            this.showNotification('Error loading models', 'error');
        }
    }

    async handleModelSelection(configPath) {
        console.log('[DEBUG] handleModelSelection() called with configPath:', configPath);
        
        if (!configPath) {
            console.log('[DEBUG] No configPath provided, hiding model info');
            this.hideModelInfo();
            return;
        }

        try {
            console.log('[DEBUG] Calling eel.get_model_details() with configPath:', configPath);
            const modelDetails = await eel.get_model_details(configPath)();
            console.log('[DEBUG] get_model_details() returned:', modelDetails);
            this.displayModelInfo(modelDetails);
        } catch (error) {
            console.error('[DEBUG] Error in handleModelSelection:', error);
            this.showNotification('Error loading model details', 'error');
        }
    }

    displayModelInfo(modelDetails) {
        const section = document.getElementById('model-info-section');
        const content = document.getElementById('model-info-content');
        
        if (!section || !content) return;

        const features = modelDetails.Config?.features || [];
        const modelName = modelDetails.model_name || 'Unknown';
        const modelType = modelDetails.Type || 'Unknown';

        content.innerHTML = `
            <div class="model-info-item">
                <span class="model-info-label">Model Name:</span>
                <span class="model-info-value">${modelName}</span>
            </div>
            <div class="model-info-item">
                <span class="model-info-label">Model Type:</span>
                <span class="model-info-value">${modelType}</span>
            </div>
            <div class="model-info-item">
                <span class="model-info-label">Required Features:</span>
                <div class="features-list">
                    ${features.map(feature => `<span class="feature-tag">${feature}</span>`).join('')}
                </div>
            </div>
        `;

        section.style.display = 'block';
    }

    hideModelInfo() {
        const section = document.getElementById('model-info-section');
        if (section) {
            section.style.display = 'none';
        }
    }

    async startAPIServer() {
        console.log('[DEBUG] startAPIServer() called');
        
        const modelSelect = document.getElementById('api-model');
        const hostInput = document.getElementById('api-host');
        const portInput = document.getElementById('api-port');
        const titleInput = document.getElementById('api-title');
        const descriptionInput = document.getElementById('api-description');

        console.log('[DEBUG] Form elements found:');
        console.log('  - modelSelect:', modelSelect);
        console.log('  - hostInput:', hostInput);
        console.log('  - portInput:', portInput);
        console.log('  - titleInput:', titleInput);
        console.log('  - descriptionInput:', descriptionInput);

        // Validation
        if (!modelSelect.value) {
            console.log('[DEBUG] No model selected, showing error');
            this.showNotification('Please select a model', 'error');
            return;
        }

        const host = hostInput.value.trim();
        const port = parseInt(portInput.value);
        const title = titleInput.value.trim();
        const description = descriptionInput.value.trim();

        console.log('[DEBUG] Processed form values:');
        console.log('  - modelSelect.value:', modelSelect.value);
        console.log('  - host:', host);
        console.log('  - port:', port);
        console.log('  - title:', title);
        console.log('  - description:', description);

        if (!host || !port || !title || !description) {
            console.log('[DEBUG] Missing required fields, showing error');
            this.showNotification('Please fill in all API settings', 'error');
            return;
        }

        try {
            console.log('[DEBUG] Disabling start button and adding loading class');
            const startBtn = document.getElementById('start-api-btn');
            startBtn.disabled = true;
            startBtn.classList.add('loading');

            // Start the API server
            const config = {
                model_path: modelSelect.value,
                host: host,
                port: port,
                title: title,
                description: description
            };
            
            console.log('[DEBUG] Calling eel.start_api_server() with config:', config);
            const result = await eel.start_api_server(config)();
            console.log('[DEBUG] start_api_server() returned:', result);

            if (result.success) {
                console.log('[DEBUG] API server started successfully, updating UI');
                this.currentAPI = {
                    model_path: modelSelect.value,
                    host: host,
                    port: port,
                    title: title,
                    description: description,
                    start_time: new Date().toISOString()
                };
                console.log('[DEBUG] currentAPI set to:', this.currentAPI);

                this.showAPIStatus();
                this.startStatusUpdates();
                this.startLogUpdates();
                this.showNotification('API server started successfully!', 'success');
            } else {
                console.log('[DEBUG] API server start failed:', result.error);
                this.showNotification(`Failed to start API server: ${result.error}`, 'error');
            }

        } catch (error) {
            console.error('[DEBUG] Error in startAPIServer:', error);
            this.showNotification('Failed to start API server', 'error');
        } finally {
            console.log('[DEBUG] Re-enabling start button');
            const startBtn = document.getElementById('start-api-btn');
            startBtn.disabled = false;
            startBtn.classList.remove('loading');
        }
    }

    async stopAPIServer() {
        try {
            const stopBtn = document.getElementById('stop-api-btn');
            stopBtn.disabled = true;
            stopBtn.classList.add('loading');

            const result = await eel.stop_api_server()();

            if (result.success) {
                this.currentAPI = null;
                this.stopStatusUpdates();
                this.stopLogUpdates();
                this.hideAPIStatus();
                this.showNotification('API server stopped successfully', 'success');
            } else {
                this.showNotification(`Failed to stop API server: ${result.error}`, 'error');
            }

        } catch (error) {
            console.error('Error stopping API server:', error);
            this.showNotification('Failed to stop API server', 'error');
        } finally {
            const stopBtn = document.getElementById('stop-api-btn');
            stopBtn.disabled = false;
            stopBtn.classList.remove('loading');
        }
    }

    showAPIStatus() {
        const section = document.getElementById('api-status-section');
        const startBtn = document.getElementById('start-api-btn');
        const stopBtn = document.getElementById('stop-api-btn');
        const testBtn = document.getElementById('test-api-btn');
        const viewDocsBtn = document.getElementById('view-docs-btn');

        if (section) {
            section.style.display = 'block';
            section.scrollIntoView({ behavior: 'smooth' });
        }

        if (startBtn) startBtn.style.display = 'none';
        if (stopBtn) stopBtn.style.display = 'inline-flex';
        if (testBtn) testBtn.style.display = 'inline-flex';
        if (viewDocsBtn) viewDocsBtn.style.display = 'inline-flex';

        this.updateStatusDisplay();
    }

    hideAPIStatus() {
        const section = document.getElementById('api-status-section');
        const startBtn = document.getElementById('start-api-btn');
        const stopBtn = document.getElementById('stop-api-btn');
        const testBtn = document.getElementById('test-api-btn');
        const viewDocsBtn = document.getElementById('view-docs-btn');
        const testingSection = document.getElementById('api-testing-section');
        // Keep logs section visible

        if (section) section.style.display = 'none';
        if (startBtn) startBtn.style.display = 'inline-flex';
        if (stopBtn) stopBtn.style.display = 'none';
        if (testBtn) testBtn.style.display = 'none';
        if (viewDocsBtn) viewDocsBtn.style.display = 'none';
        if (testingSection) testingSection.style.display = 'none';
    }

    updateStatusDisplay() {
        if (!this.currentAPI) return;

        const statusBadge = document.getElementById('api-status');
        const modelNameBadge = document.getElementById('api-model-name');
        const statusMessage = document.getElementById('api-status-message');
        const apiUrl = document.getElementById('api-url');
        const docsUrl = document.getElementById('api-docs-url');
        const startTime = document.getElementById('api-start-time');
        const uptime = document.getElementById('api-uptime');

        if (statusBadge) {
            statusBadge.textContent = 'Running';
            statusBadge.className = 'status-badge running';
        }

        if (modelNameBadge) {
            modelNameBadge.textContent = this.currentAPI.title;
        }

        if (statusMessage) {
            statusMessage.textContent = 'API server is running and ready to accept requests';
        }

        if (apiUrl) {
            apiUrl.textContent = `http://${this.currentAPI.host}:${this.currentAPI.port}`;
        }

        if (docsUrl) {
            docsUrl.textContent = `http://${this.currentAPI.host}:${this.currentAPI.port}/docs`;
        }

        if (startTime) {
            startTime.textContent = new Date(this.currentAPI.start_time).toLocaleString();
        }

        if (uptime) {
            this.updateUptime();
        }
    }

    updateUptime() {
        if (!this.currentAPI) return;

        const uptimeElement = document.getElementById('api-uptime');
        if (!uptimeElement) return;

        const startTime = new Date(this.currentAPI.start_time);
        const currentTime = new Date();
        const duration = currentTime - startTime;

        uptimeElement.textContent = this.formatDuration(duration);
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

    startStatusUpdates() {
        if (this.statusUpdateInterval) {
            clearInterval(this.statusUpdateInterval);
        }

        this.statusUpdateInterval = setInterval(() => {
            this.updateUptime();
        }, 1000);
    }

    stopStatusUpdates() {
        if (this.statusUpdateInterval) {
            clearInterval(this.statusUpdateInterval);
            this.statusUpdateInterval = null;
        }
    }

    startLogUpdates() {
        if (this.logUpdateInterval) {
            clearInterval(this.logUpdateInterval);
        }

        this.logUpdateInterval = setInterval(async () => {
            try {
                const logs = await eel.get_api_logs()();
                this.updateLogsDisplay(logs);
            } catch (error) {
                console.error('Error fetching API logs:', error);
            }
        }, 1000); // Update every second
    }

    stopLogUpdates() {
        if (this.logUpdateInterval) {
            clearInterval(this.logUpdateInterval);
            this.logUpdateInterval = null;
        }
    }

    updateLogsDisplay(logs) {
        const logsContainer = document.getElementById('api-logs');
        if (!logsContainer) return;

        // Clear existing logs
        logsContainer.innerHTML = '';

        // Add each log entry
        logs.forEach(log => {
            const logEntry = document.createElement('div');
            logEntry.className = 'log-entry';
            
            // Parse timestamp and message
            const timestampMatch = log.match(/^\[(\d{2}:\d{2}:\d{2})\]/);
            const timestamp = timestampMatch ? timestampMatch[1] : '';
            const message = log.replace(/^\[\d{2}:\d{2}:\d{2}\]\s*/, '');
            
            logEntry.innerHTML = `
                <span class="log-timestamp">[${timestamp}]</span>
                <span class="log-message">${message}</span>
            `;
            
            logsContainer.appendChild(logEntry);
        });

        // Auto-scroll if enabled
        if (this.autoScroll) {
            logsContainer.scrollTop = logsContainer.scrollHeight;
        }
    }

    toggleTestingSection() {
        const section = document.getElementById('api-testing-section');
        if (section) {
            const isVisible = section.style.display !== 'none';
            section.style.display = isVisible ? 'none' : 'block';
            
            if (!isVisible) {
                section.scrollIntoView({ behavior: 'smooth' });
            }
        }
    }

    async sendTestRequest() {
        const featuresTextarea = document.getElementById('test-features');
        const resultsSection = document.getElementById('test-results');
        const responseDiv = document.getElementById('test-response');

        if (!featuresTextarea || !resultsSection || !responseDiv) return;

        const featuresText = featuresTextarea.value.trim();
        if (!featuresText) {
            this.showNotification('Please enter test features', 'error');
            return;
        }

        try {
            const features = JSON.parse(featuresText);
            
            const result = await eel.test_api_prediction({
                features: features
            })();

            if (result.success) {
                responseDiv.textContent = JSON.stringify(result.data, null, 2);
                responseDiv.className = 'test-response success';
                resultsSection.style.display = 'block';
                this.showNotification('Test request successful!', 'success');
            } else {
                responseDiv.textContent = `Error: ${result.error}`;
                responseDiv.className = 'test-response error';
                resultsSection.style.display = 'block';
                this.showNotification(`Test request failed: ${result.error}`, 'error');
            }

        } catch (error) {
            if (error instanceof SyntaxError) {
                this.showNotification('Invalid JSON format in test features', 'error');
            } else {
                console.error('Test request error:', error);
                this.showNotification('Test request failed', 'error');
            }
        }
    }

    clearTestResults() {
        const featuresTextarea = document.getElementById('test-features');
        const resultsSection = document.getElementById('test-results');
        const responseDiv = document.getElementById('test-response');

        if (featuresTextarea) featuresTextarea.value = '';
        if (resultsSection) resultsSection.style.display = 'none';
        if (responseDiv) {
            responseDiv.textContent = '';
            responseDiv.className = 'test-response';
        }
    }

    openAPIDocs() {
        if (this.currentAPI) {
            const docsUrl = `http://${this.currentAPI.host}:${this.currentAPI.port}/docs`;
            window.open(docsUrl, '_blank');
        }
    }

    async copyAPIUrl() {
        if (this.currentAPI) {
            const apiUrl = `http://${this.currentAPI.host}:${this.currentAPI.port}`;
            try {
                await navigator.clipboard.writeText(apiUrl);
                this.showNotification('API URL copied to clipboard!', 'success');
            } catch (error) {
                console.error('Failed to copy URL:', error);
                this.showNotification('Failed to copy URL', 'error');
            }
        }
    }

    async clearLogs() {
        try {
            const result = await eel.clear_api_logs()();
            if (result.success) {
                this.showNotification('API logs cleared', 'success');
            } else {
                this.showNotification('Failed to clear logs', 'error');
            }
        } catch (error) {
            console.error('Error clearing logs:', error);
            this.showNotification('Failed to clear logs', 'error');
        }
    }

    exportLogs() {
        const logsContainer = document.getElementById('api-logs');
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
        a.download = `api-logs-${new Date().toISOString().split('T')[0]}.txt`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }

    toggleAutoScroll() {
        this.autoScroll = !this.autoScroll;
        const toggleBtn = document.getElementById('toggle-api-auto-scroll');
        if (toggleBtn) {
            toggleBtn.classList.toggle('active', this.autoScroll);
        }
    }

    async loadAPIHistory() {
        // This would load from a persistent storage or API
        // For now, we'll show a placeholder
        const historyContainer = document.getElementById('api-history');
        if (historyContainer) {
            historyContainer.innerHTML = `
                <div class="history-item">
                    <div class="history-header">
                        <div class="history-title">API History</div>
                        <div class="history-status completed">Completed</div>
                    </div>
                    <div class="history-details">
                        <div class="history-detail">
                            <div class="history-detail-label">Model</div>
                            <div class="history-detail-value">NN_300T_NQ_SuperCCI</div>
                        </div>
                        <div class="history-detail">
                            <div class="history-detail-label">Port</div>
                            <div class="history-detail-value">8000</div>
                        </div>
                        <div class="history-detail">
                            <div class="history-detail-label">Duration</div>
                            <div class="history-detail-value">2h 15m</div>
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

    showNotification(message, type = 'info') {
        // Create notification element
        const notification = document.createElement('div');
        notification.className = `notification notification-${type}`;
        notification.innerHTML = `
            <span class="notification-message">${message}</span>
            <button class="notification-close">&times;</button>
        `;
        
        // Add to page
        document.body.appendChild(notification);
        
        // Show notification
        setTimeout(() => notification.classList.add('show'), 100);
        
        // Auto-remove after 5 seconds
        setTimeout(() => {
            notification.classList.remove('show');
            setTimeout(() => notification.remove(), 300);
        }, 5000);
        
        // Close button functionality
        notification.querySelector('.notification-close').addEventListener('click', () => {
            notification.classList.remove('show');
            setTimeout(() => notification.remove(), 300);
        });
    }

    static loadAPIHosting() {
        console.log('Loading API Hosting tab...');
        // Initialize API hosting functionality when tab is loaded
        if (!window.apiHostingInstance) {
            window.apiHostingInstance = new APIHosting();
        }
    }
}

// Initialize API hosting when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.APIHosting = new APIHosting();
});

// Add a method to handle tab switching
window.APIHostingTab = {
    loadContent: function() {
        if (window.APIHosting) {
            window.APIHosting.loadAvailableModels();
            window.APIHosting.loadAPIHistory();
        }
    }
};
