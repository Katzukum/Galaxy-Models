/**
 * Ensemble Training Tab JavaScript
 * Handles ensemble model training functionality
 */

// Global variables for ensemble training
let availableModels = [];
let selectedModels = [];
let ensembleTrainingInProgress = false;

// Note: Ensemble training is initialized when the tab is loaded, not on DOM ready

// Fallback: Initialize if the ensemble training tab is visible on page load
document.addEventListener('DOMContentLoaded', function() {
    console.log('[ENSEMBLE] DOM loaded, checking for ensemble training tab...');
    
    // Check if ensemble training tab is visible (for direct access)
    const ensembleTab = document.getElementById('ensemble-training-content');
    console.log('[ENSEMBLE] Ensemble tab element:', ensembleTab);
    
    if (ensembleTab) {
        console.log('[ENSEMBLE] Ensemble tab found, checking if active...');
        console.log('[ENSEMBLE] Tab classes:', ensembleTab.classList.toString());
        console.log('[ENSEMBLE] Is active:', ensembleTab.classList.contains('active'));
        
        if (ensembleTab.classList.contains('active')) {
            console.log('[ENSEMBLE] Ensemble training tab is active on page load, initializing...');
            initializeEnsembleTraining();
        } else {
            console.log('[ENSEMBLE] Ensemble training tab not active on page load');
        }
    } else {
        console.error('[ENSEMBLE] Ensemble training tab element not found!');
    }
    
    // Make function globally available
    window.initializeEnsembleTraining = initializeEnsembleTraining;
    console.log('[ENSEMBLE] Made initializeEnsembleTraining globally available');
});

/**
 * Initialize ensemble training functionality
 */
function initializeEnsembleTraining() {
    console.log('Initializing ensemble training...');
    
    // Check if we're on the ensemble training tab
    const ensembleTab = document.getElementById('ensemble-training-content');
    if (!ensembleTab) {
        console.error('Ensemble training tab not found');
        return;
    }
    
    console.log('Ensemble training tab found, initializing...');
    
    // Load available models
    loadAvailableModels();
    
    // Setup event listeners
    setupEnsembleEventListeners();
    
    // Initialize form
    initializeEnsembleForm();
    
    console.log('Ensemble training initialization completed');
}

/**
 * Setup event listeners for ensemble training
 */
function setupEnsembleEventListeners() {
    // Ensemble type change
    const ensembleTypeSelect = document.getElementById('ensemble-type');
    if (ensembleTypeSelect) {
        ensembleTypeSelect.addEventListener('change', handleEnsembleTypeChange);
    }
    
    // Advanced options toggle
    const advancedOptionsCheckbox = document.getElementById('ensemble-advanced-options');
    if (advancedOptionsCheckbox) {
        advancedOptionsCheckbox.addEventListener('change', toggleAdvancedOptions);
    }
    
    // Model selection buttons
    const selectAllBtn = document.getElementById('select-all-models');
    const clearBtn = document.getElementById('clear-selection');
    
    if (selectAllBtn) {
        selectAllBtn.addEventListener('click', selectAllModels);
    }
    
    if (clearBtn) {
        clearBtn.addEventListener('click', clearModelSelection);
    }
    
    // Form submission
    const ensembleForm = document.getElementById('ensemble-training-form');
    if (ensembleForm) {
        ensembleForm.addEventListener('submit', handleEnsembleFormSubmit);
    }
    
    // Reset form
    const resetBtn = document.getElementById('reset-ensemble-form');
    if (resetBtn) {
        resetBtn.addEventListener('click', resetEnsembleForm);
    }
    
    // File input change
    const csvFileInput = document.getElementById('ensemble-csv-file');
    if (csvFileInput) {
        csvFileInput.addEventListener('change', handleCsvFileChange);
    }
}

/**
 * Load available models from the backend
 */
async function loadAvailableModels() {
    console.log('[ENSEMBLE] loadAvailableModels called');
    
    try {
        console.log('[ENSEMBLE] Starting model loading process...');
        
        // Check if eel is available
        console.log('[ENSEMBLE] Checking EEL availability...');
        console.log('[ENSEMBLE] typeof eel:', typeof eel);
        console.log('[ENSEMBLE] eel object:', eel);
        
        if (typeof eel === 'undefined') {
            console.error('[ENSEMBLE] EEL is not available');
            showModelSelectionError('EEL connection not available. Please refresh the page.');
            return;
        }
        
        // Check if get_models function exists
        console.log('[ENSEMBLE] Checking get_models function...');
        console.log('[ENSEMBLE] typeof eel.get_models:', typeof eel.get_models);
        console.log('[ENSEMBLE] eel.get_models:', eel.get_models);
        
        if (typeof eel.get_models !== 'function') {
            console.error('[ENSEMBLE] get_models function not available');
            console.log('[ENSEMBLE] Available eel functions:', Object.keys(eel));
            showModelSelectionError('get_models function not available. Please check the backend.');
            return;
        }
        
        console.log('[ENSEMBLE] Calling eel.get_models()...');
        const models = await eel.get_models()();
        console.log('[ENSEMBLE] Received models:', models);
        console.log('[ENSEMBLE] Models type:', typeof models);
        console.log('[ENSEMBLE] Models length:', models ? models.length : 'undefined');
        
        if (models && models.length > 0) {
            availableModels = models;
            console.log('[ENSEMBLE] Setting availableModels:', availableModels);
            renderModelSelectionList();
            console.log(`[ENSEMBLE] Loaded ${models.length} models successfully`);
        } else {
            console.log('[ENSEMBLE] No models found or empty array');
            showModelSelectionError('No trained models found. Please train some models first.');
        }
    } catch (error) {
        console.error('[ENSEMBLE] Error loading models:', error);
        console.error('[ENSEMBLE] Error stack:', error.stack);
        showModelSelectionError('Failed to load models: ' + error.message);
    }
}

/**
 * Render the model selection list
 */
function renderModelSelectionList() {
    console.log('[ENSEMBLE] renderModelSelectionList called');
    console.log('[ENSEMBLE] availableModels:', availableModels);
    console.log('[ENSEMBLE] availableModels.length:', availableModels.length);
    
    const modelListContainer = document.getElementById('model-selection-list');
    console.log('[ENSEMBLE] modelListContainer:', modelListContainer);
    
    if (!modelListContainer) {
        console.error('[ENSEMBLE] model-selection-list element not found!');
        return;
    }
    
    if (availableModels.length === 0) {
        console.log('[ENSEMBLE] No models available, showing no-models message');
        modelListContainer.innerHTML = '<div class="no-models">No trained models available</div>';
        return;
    }
    
    console.log('[ENSEMBLE] Rendering model selection list with', availableModels.length, 'models');
    
    // Group models by type
    const modelsByType = groupModelsByType(availableModels);
    
    let html = '';
    for (const [type, models] of Object.entries(modelsByType)) {
        html += `<div class="model-type-group">
            <div class="model-type-header">
                <h4>${getModelTypeIcon(type)} ${type}</h4>
                <span class="model-count">${models.length} model${models.length !== 1 ? 's' : ''}</span>
            </div>
            <div class="model-type-models">`;
        
        models.forEach(model => {
            const isSelected = selectedModels.some(selected => selected.name === model.name);
            html += `<div class="model-item ${isSelected ? 'selected' : ''}" data-model-name="${model.name}">
                <label class="model-checkbox-label">
                    <input type="checkbox" class="model-checkbox" value="${model.name}" ${isSelected ? 'checked' : ''}>
                    <span class="checkmark"></span>
                    <div class="model-info">
                        <div class="model-name">${model.name}</div>
                        <div class="model-type">${model.type}</div>
                    </div>
                </label>
            </div>`;
        });
        
        html += '</div></div>';
    }
    
    modelListContainer.innerHTML = html;
    
    // Add event listeners to checkboxes
    const checkboxes = modelListContainer.querySelectorAll('.model-checkbox');
    checkboxes.forEach(checkbox => {
        checkbox.addEventListener('change', handleModelSelectionChange);
    });
}

/**
 * Group models by type
 */
function groupModelsByType(models) {
    const grouped = {};
    models.forEach(model => {
        if (!grouped[model.type]) {
            grouped[model.type] = [];
        }
        grouped[model.type].push(model);
    });
    return grouped;
}

/**
 * Get icon for model type
 */
function getModelTypeIcon(type) {
    const icons = {
        'XGBoost': '🌳',
        'Neural Network': '🧠',
        'Transformer': '🔄',
        'PPO Agent': '🤖'
    };
    return icons[type] || '🤖';
}

/**
 * Handle model selection change
 */
function handleModelSelectionChange(event) {
    const modelName = event.target.value;
    const isSelected = event.target.checked;
    
    if (isSelected) {
        // Add model to selection
        const model = availableModels.find(m => m.name === modelName);
        if (model && !selectedModels.some(m => m.name === modelName)) {
            selectedModels.push(model);
        }
    } else {
        // Remove model from selection
        selectedModels = selectedModels.filter(m => m.name !== modelName);
    }
    
    // Update UI
    updateModelSelectionUI();
    updateWeightConfiguration();
    validateEnsembleForm();
}

/**
 * Update model selection UI
 */
function updateModelSelectionUI() {
    // Update selected count
    const selectedCount = selectedModels.length;
    const modelSelectionHeader = document.querySelector('.model-selection-header span');
    if (modelSelectionHeader) {
        modelSelectionHeader.textContent = `Available Models (${selectedCount} selected)`;
    }
    
    // Update model items
    const modelItems = document.querySelectorAll('.model-item');
    modelItems.forEach(item => {
        const modelName = item.dataset.modelName;
        const isSelected = selectedModels.some(m => m.name === modelName);
        item.classList.toggle('selected', isSelected);
    });
}

/**
 * Select all models
 */
function selectAllModels() {
    selectedModels = [...availableModels];
    
    // Update checkboxes
    const checkboxes = document.querySelectorAll('.model-checkbox');
    checkboxes.forEach(checkbox => {
        checkbox.checked = true;
    });
    
    updateModelSelectionUI();
    updateWeightConfiguration();
    validateEnsembleForm();
}

/**
 * Clear model selection
 */
function clearModelSelection() {
    selectedModels = [];
    
    // Update checkboxes
    const checkboxes = document.querySelectorAll('.model-checkbox');
    checkboxes.forEach(checkbox => {
        checkbox.checked = false;
    });
    
    updateModelSelectionUI();
    updateWeightConfiguration();
    validateEnsembleForm();
}

/**
 * Handle ensemble type change
 */
function handleEnsembleTypeChange(event) {
    const ensembleType = event.target.value;
    
    // Show/hide weight configuration
    const weightConfig = document.getElementById('weight-config');
    const metaLearnerConfig = document.getElementById('meta-learner-config');
    
    if (weightConfig) {
        weightConfig.style.display = ensembleType === 'weighted' ? 'block' : 'none';
    }
    
    if (metaLearnerConfig) {
        metaLearnerConfig.style.display = ensembleType === 'stacking' ? 'block' : 'none';
    }
    
    // Update help text
    updateEnsembleHelpText(ensembleType);
    
    // Update weight configuration
    updateWeightConfiguration();
    
    // Validate form
    validateEnsembleForm();
}

/**
 * Update ensemble help text
 */
function updateEnsembleHelpText(ensembleType) {
    const helpTexts = document.querySelectorAll('.help-text[data-ensemble]');
    helpTexts.forEach(helpText => {
        helpText.style.display = helpText.dataset.ensemble === ensembleType ? 'block' : 'none';
    });
}

/**
 * Update weight configuration
 */
function updateWeightConfiguration() {
    const weightConfig = document.getElementById('weight-config');
    const weightInputs = document.getElementById('weight-inputs');
    
    if (!weightConfig || !weightInputs) return;
    
    if (selectedModels.length === 0) {
        weightInputs.innerHTML = '<div class="no-models">Select models first</div>';
        return;
    }
    
    // Calculate default weights
    const defaultWeight = 1.0 / selectedModels.length;
    
    let html = '<div class="weight-inputs-grid">';
    selectedModels.forEach((model, index) => {
        html += `<div class="weight-input-group">
            <label for="weight-${index}">${model.name}</label>
            <input type="number" id="weight-${index}" class="weight-input" 
                   value="${defaultWeight.toFixed(3)}" min="0" max="1" step="0.001">
        </div>`;
    });
    html += '</div>';
    
    // Add weight validation
    html += '<div class="weight-validation">
        <span id="weight-sum">Sum: 1.000</span>
        <span id="weight-status" class="weight-status valid">✓ Valid</span>
    </div>';
    
    weightInputs.innerHTML = html;
    
    // Add event listeners to weight inputs
    const weightInputsElements = weightInputs.querySelectorAll('.weight-input');
    weightInputsElements.forEach(input => {
        input.addEventListener('input', validateWeights);
    });
    
    // Initial validation
    validateWeights();
}

/**
 * Validate weights
 */
function validateWeights() {
    const weightInputs = document.querySelectorAll('.weight-input');
    const weightSumElement = document.getElementById('weight-sum');
    const weightStatusElement = document.getElementById('weight-status');
    
    if (!weightInputs.length || !weightSumElement || !weightStatusElement) return;
    
    let sum = 0;
    weightInputs.forEach(input => {
        sum += parseFloat(input.value) || 0;
    });
    
    const isValid = Math.abs(sum - 1.0) < 0.001;
    
    weightSumElement.textContent = `Sum: ${sum.toFixed(3)}`;
    weightStatusElement.textContent = isValid ? '✓ Valid' : '✗ Invalid';
    weightStatusElement.className = `weight-status ${isValid ? 'valid' : 'invalid'}`;
    
    // Update form validation
    validateEnsembleForm();
}

/**
 * Toggle advanced options
 */
function toggleAdvancedOptions(event) {
    const advancedPanel = document.getElementById('ensemble-advanced-panel');
    if (advancedPanel) {
        advancedPanel.style.display = event.target.checked ? 'block' : 'none';
    }
}

/**
 * Handle CSV file change
 */
function handleCsvFileChange(event) {
    const file = event.target.files[0];
    if (file) {
        console.log('CSV file selected:', file.name);
        validateEnsembleForm();
    }
}

/**
 * Validate ensemble form
 */
function validateEnsembleForm() {
    const ensembleType = document.getElementById('ensemble-type').value;
    const ensembleName = document.getElementById('ensemble-name').value;
    const csvFile = document.getElementById('ensemble-csv-file').files[0];
    
    let isValid = true;
    let errors = [];
    
    // Check ensemble type
    if (!ensembleType) {
        isValid = false;
        errors.push('Please select an ensemble type');
    }
    
    // Check model selection
    if (selectedModels.length < 2) {
        isValid = false;
        errors.push('Please select at least 2 models');
    }
    
    // Check ensemble name
    if (!ensembleName.trim()) {
        isValid = false;
        errors.push('Please enter an ensemble name');
    }
    
    // Check CSV file
    if (!csvFile) {
        isValid = false;
        errors.push('Please select a CSV file');
    }
    
    // Check weights for weighted ensemble
    if (ensembleType === 'weighted') {
        const weightStatus = document.getElementById('weight-status');
        if (weightStatus && !weightStatus.classList.contains('valid')) {
            isValid = false;
            errors.push('Please ensure weights sum to 1.0');
        }
    }
    
    // Update submit button
    const submitBtn = document.getElementById('start-ensemble-training');
    if (submitBtn) {
        submitBtn.disabled = !isValid;
        submitBtn.textContent = isValid ? '🚀 Start Ensemble Training' : '❌ Fix Errors First';
    }
    
    return isValid;
}

/**
 * Handle ensemble form submission
 */
async function handleEnsembleFormSubmit(event) {
    event.preventDefault();
    
    if (!validateEnsembleForm()) {
        showNotification('Please fix all errors before submitting', 'error');
        return;
    }
    
    if (ensembleTrainingInProgress) {
        showNotification('Ensemble training is already in progress', 'warning');
        return;
    }
    
    try {
        // Get form data
        const formData = getEnsembleFormData();
        
        // Start ensemble training
        await startEnsembleTraining(formData);
        
    } catch (error) {
        console.error('Error starting ensemble training:', error);
        showNotification('Failed to start ensemble training: ' + error.message, 'error');
    }
}

/**
 * Get ensemble form data
 */
function getEnsembleFormData() {
    const ensembleType = document.getElementById('ensemble-type').value;
    const ensembleName = document.getElementById('ensemble-name').value;
    const csvFile = document.getElementById('ensemble-csv-file').files[0];
    
    // Get weights for weighted ensemble
    let weights = null;
    if (ensembleType === 'weighted') {
        weights = {};
        selectedModels.forEach((model, index) => {
            const weightInput = document.getElementById(`weight-${index}`);
            weights[model.name] = parseFloat(weightInput.value);
        });
    }
    
    // Get advanced options
    const advancedOptions = {
        cvFolds: parseInt(document.getElementById('ensemble-cv-folds').value),
        metaLearner: document.getElementById('meta-learner').value,
        validationSplit: parseFloat(document.getElementById('ensemble-validation-split').value),
        randomState: parseInt(document.getElementById('ensemble-random-state').value)
    };
    
    return {
        ensembleType,
        ensembleName,
        selectedModels: selectedModels.map(m => ({
            name: m.name,
            type: m.type,
            configPath: m.config_path
        })),
        weights,
        csvFile,
        advancedOptions
    };
}

/**
 * Start ensemble training
 */
async function startEnsembleTraining(formData) {
    console.log('Starting ensemble training:', formData);
    
    ensembleTrainingInProgress = true;
    
    try {
        // Show progress panel
        showEnsembleTrainingProgress();
        
        // Update status
        updateEnsembleTrainingStatus('Preparing ensemble training...', 0);
        
        // Start training process
        const trainingId = await eel.start_ensemble_training(
            formData.ensembleType,
            formData.ensembleName,
            formData.selectedModels,
            formData.weights,
            formData.advancedOptions
        )();
        
        console.log('Ensemble training started with ID:', trainingId);
        
        // Monitor training progress
        monitorEnsembleTrainingProgress(trainingId);
        
    } catch (error) {
        console.error('Error starting ensemble training:', error);
        showNotification('Failed to start ensemble training: ' + error.message, 'error');
        ensembleTrainingInProgress = false;
        hideEnsembleTrainingProgress();
    }
}

/**
 * Show ensemble training progress
 */
function showEnsembleTrainingProgress() {
    const progressCard = document.getElementById('ensemble-training-progress');
    if (progressCard) {
        progressCard.style.display = 'block';
    }
}

/**
 * Hide ensemble training progress
 */
function hideEnsembleTrainingProgress() {
    const progressCard = document.getElementById('ensemble-training-progress');
    if (progressCard) {
        progressCard.style.display = 'none';
    }
}

/**
 * Update ensemble training status
 */
function updateEnsembleTrainingStatus(status, progress) {
    const statusValue = document.getElementById('ensemble-status-value');
    const progressValue = document.getElementById('ensemble-progress-value');
    const progressFill = document.getElementById('ensemble-progress-fill');
    const progressText = document.getElementById('ensemble-progress-text');
    
    if (statusValue) statusValue.textContent = status;
    if (progressValue) progressValue.textContent = `${progress}%`;
    if (progressFill) progressFill.style.width = `${progress}%`;
    if (progressText) progressText.textContent = `${progress}%`;
}

/**
 * Monitor ensemble training progress
 */
async function monitorEnsembleTrainingProgress(trainingId) {
    try {
        const status = await eel.get_training_status(trainingId)();
        
        if (status) {
            updateEnsembleTrainingStatus(status.message, status.progress);
            
            if (status.status === 'completed') {
                console.log('Ensemble training completed');
                showNotification('Ensemble training completed successfully!', 'success');
                ensembleTrainingInProgress = false;
                loadEnsembleTrainingHistory();
            } else if (status.status === 'failed' || status.status === 'error') {
                console.error('Ensemble training failed:', status.message);
                showNotification('Ensemble training failed: ' + status.message, 'error');
                ensembleTrainingInProgress = false;
            } else {
                // Continue monitoring
                setTimeout(() => monitorEnsembleTrainingProgress(trainingId), 2000);
            }
        }
    } catch (error) {
        console.error('Error monitoring ensemble training:', error);
        ensembleTrainingInProgress = false;
    }
}

/**
 * Load ensemble training history
 */
async function loadEnsembleTrainingHistory() {
    try {
        const history = await eel.get_ensemble_training_history()();
        renderEnsembleTrainingHistory(history);
    } catch (error) {
        console.error('Error loading ensemble training history:', error);
    }
}

/**
 * Render ensemble training history
 */
function renderEnsembleTrainingHistory(history) {
    const historyContainer = document.getElementById('ensemble-training-history');
    if (!historyContainer) return;
    
    if (!history || history.length === 0) {
        historyContainer.innerHTML = '<div class="no-history">No ensemble training history found</div>';
        return;
    }
    
    let html = '<div class="training-history-list">';
    history.forEach(entry => {
        html += `<div class="training-history-item">
            <div class="history-header">
                <h4>${entry.ensembleName}</h4>
                <span class="history-status ${entry.status}">${entry.status}</span>
            </div>
            <div class="history-details">
                <div class="history-info">
                    <span><strong>Type:</strong> ${entry.ensembleType}</span>
                    <span><strong>Models:</strong> ${entry.modelCount}</span>
                    <span><strong>Created:</strong> ${new Date(entry.createdAt).toLocaleString()}</span>
                </div>
                <div class="history-actions">
                    <button class="btn-secondary" onclick="viewEnsembleDetails('${entry.ensembleName}')">View Details</button>
                    <button class="btn-secondary" onclick="downloadEnsembleModel('${entry.ensembleName}')">Download</button>
                </div>
            </div>
        </div>`;
    });
    html += '</div>';
    
    historyContainer.innerHTML = html;
}

/**
 * Reset ensemble form
 */
function resetEnsembleForm() {
    // Reset form fields
    document.getElementById('ensemble-type').value = '';
    document.getElementById('ensemble-name').value = '';
    document.getElementById('ensemble-csv-file').value = '';
    
    // Clear model selection
    clearModelSelection();
    
    // Hide advanced options
    document.getElementById('ensemble-advanced-options').checked = false;
    toggleAdvancedOptions({ target: { checked: false } });
    
    // Hide progress
    hideEnsembleTrainingProgress();
    
    // Reset validation
    validateEnsembleForm();
    
    console.log('Ensemble form reset');
}

/**
 * Initialize ensemble form
 */
function initializeEnsembleForm() {
    // Set up form validation
    validateEnsembleForm();
    
    // Load training history
    loadEnsembleTrainingHistory();
}

/**
 * Show model selection error
 */
function showModelSelectionError(message) {
    console.log('[ENSEMBLE] showModelSelectionError called with message:', message);
    
    const modelListContainer = document.getElementById('model-selection-list');
    console.log('[ENSEMBLE] modelListContainer for error:', modelListContainer);
    
    if (modelListContainer) {
        modelListContainer.innerHTML = `<div class="error-message">${message}</div>`;
        console.log('[ENSEMBLE] Error message displayed');
    } else {
        console.error('[ENSEMBLE] model-selection-list element not found for error display');
    }
}

/**
 * Show notification
 */
function showNotification(message, type = 'info') {
    // This would integrate with your existing notification system
    console.log(`[${type.toUpperCase()}] ${message}`);
}

// Export functions for global access
window.viewEnsembleDetails = function(ensembleName) {
    console.log('View ensemble details:', ensembleName);
    // Implement ensemble details view
};

window.downloadEnsembleModel = function(ensembleName) {
    console.log('Download ensemble model:', ensembleName);
    // Implement ensemble model download
};