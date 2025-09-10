/**
 * Ensemble Training Tab JavaScript
 * Handles ensemble model training functionality
 */

// Global variables for ensemble training
let availableModels = [];
let selectedModels = [];
let ensembleTrainingInProgress = false;
let ensembleInitialized = false;

// Note: Ensemble training is initialized when the tab is loaded, not on DOM ready

// Initialize when DOM is loaded and watch for tab activation
document.addEventListener('DOMContentLoaded', function() {
    console.log('[ENSEMBLE] DOM loaded, setting up ensemble training...');
    
    // Make function globally available immediately
    window.initializeEnsembleTraining = initializeEnsembleTraining;
    console.log('[ENSEMBLE] Made initializeEnsembleTraining globally available');
    
    // Check if ensemble training tab is visible on page load
    const ensembleTab = document.getElementById('ensemble-training-content');
    console.log('[ENSEMBLE] Ensemble tab element:', ensembleTab);
    
    if (ensembleTab) {
        console.log('[ENSEMBLE] Ensemble tab found, checking if active...');
        console.log('[ENSEMBLE] Tab classes:', ensembleTab.classList.toString());
        console.log('[ENSEMBLE] Is active:', ensembleTab.classList.contains('active'));
        
        if (ensembleTab.classList.contains('active')) {
            console.log('[ENSEMBLE] Ensemble training tab is active on page load, initializing...');
            initializeEnsembleTraining();
        }
        
        // Set up MutationObserver to watch for tab activation
        const observer = new MutationObserver(function(mutations) {
            mutations.forEach(function(mutation) {
                if (mutation.type === 'attributes' && mutation.attributeName === 'class') {
                    const target = mutation.target;
                    if (target.id === 'ensemble-training-content' && target.classList.contains('active')) {
                        console.log('[ENSEMBLE] Ensemble tab became active, initializing...');
                        initializeEnsembleTraining();
                    }
                }
            });
        });
        
        // Start observing the ensemble tab for class changes
        observer.observe(ensembleTab, { attributes: true, attributeFilter: ['class'] });
        console.log('[ENSEMBLE] Set up observer for tab activation');
        
        // Also observe the tab content container for new active tabs
        const tabContainer = document.getElementById('tab-content-container');
        if (tabContainer) {
            const containerObserver = new MutationObserver(function(mutations) {
                mutations.forEach(function(mutation) {
                    if (mutation.type === 'attributes' && mutation.attributeName === 'class') {
                        mutation.addedNodes.forEach(function(node) {
                            if (node.nodeType === Node.ELEMENT_NODE && 
                                node.id === 'ensemble-training-content' && 
                                node.classList.contains('active')) {
                                console.log('[ENSEMBLE] Ensemble tab activated via container observer');
                                setTimeout(() => initializeEnsembleTraining(), 100);
                            }
                        });
                    }
                });
            });
            containerObserver.observe(tabContainer, { 
                attributes: true, 
                childList: true, 
                subtree: true,
                attributeFilter: ['class']
            });
        }
    } else {
        console.error('[ENSEMBLE] Ensemble training tab element not found!');
    }
});

/**
 * Initialize ensemble training functionality
 */
function initializeEnsembleTraining() {
    console.log('Initializing ensemble training...');
    
    // Prevent multiple initializations
    if (ensembleInitialized) {
        console.log('[ENSEMBLE] Already initialized, skipping...');
        return;
    }
    
    // Check if we're on the ensemble training tab
    const ensembleTab = document.getElementById('ensemble-training-content');
    if (!ensembleTab) {
        console.error('Ensemble training tab not found');
        return;
    }
    
    // Check if tab is actually active
    if (!ensembleTab.classList.contains('active')) {
        console.log('[ENSEMBLE] Tab not active, skipping initialization...');
        return;
    }
    
    console.log('Ensemble training tab found and active, initializing...');
    ensembleInitialized = true;
    
    // Load available models
    loadAvailableModels();
    
    // Load available CSV files
    loadAvailableCSVFiles();
    
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
        ensembleTypeSelect.addEventListener('change', (event) => {
            handleEnsembleTypeChange(event);
            validateEnsembleForm();
        });
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
    
    // Ensemble name input
    const ensembleNameInput = document.getElementById('ensemble-name');
    if (ensembleNameInput) {
        ensembleNameInput.addEventListener('input', validateEnsembleForm);
        ensembleNameInput.addEventListener('blur', validateEnsembleForm);
    }
    
    // File input change
    const csvFileInput = document.getElementById('ensemble-csv-file');
    if (csvFileInput) {
        csvFileInput.addEventListener('change', (event) => {
            handleCsvFileChange(event);
            validateEnsembleForm();
        });
        
        // Drag and drop functionality
        const fileDisplay = document.querySelector('#ensemble-training-content .file-input-display');
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
                    csvFileInput.files = files;
                    handleCsvFileChange({ target: csvFileInput });
                }
            });
        }
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
    html += `<div class="weight-validation">
        <span id="weight-sum">Sum: 1.000</span>
        <span id="weight-status" class="weight-status valid">✓ Valid</span>
    </div>`;
    
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
    const fileDisplay = document.querySelector('#ensemble-training-content .file-input-display');
    const fileText = document.querySelector('#ensemble-training-content .file-input-text');
    const fileIcon = document.querySelector('#ensemble-training-content .file-input-icon');

    if (file) {
        console.log('CSV file selected:', file.name);
        fileDisplay.classList.add('has-file');
        fileText.textContent = file.name;
        fileIcon.textContent = '✅';
        validateEnsembleForm();
    } else {
        fileDisplay.classList.remove('has-file');
        fileText.textContent = 'Choose CSV file or drag & drop';
        fileIcon.textContent = '📁';
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
    
    // Clear previous error indicators
    clearValidationErrors();
    
    // Check ensemble type
    if (!ensembleType) {
        isValid = false;
        errors.push('Please select an ensemble type');
        highlightFieldError('ensemble-type', 'Please select an ensemble type');
    } else {
        clearFieldError('ensemble-type');
    }
    
    // Check model selection
    if (selectedModels.length < 2) {
        isValid = false;
        errors.push('Please select at least 2 models');
        highlightFieldError('model-selection-list', 'Please select at least 2 models');
    } else {
        clearFieldError('model-selection-list');
    }
    
    // Check ensemble name
    if (!ensembleName.trim()) {
        isValid = false;
        errors.push('Please enter an ensemble name');
        highlightFieldError('ensemble-name', 'Please enter an ensemble name');
    } else {
        clearFieldError('ensemble-name');
    }
    
    // Check CSV file - either from available files list or file input
    const selectedFileItem = document.querySelector('#ensemble-available-csv-files .csv-file-item.selected');
    const fileInput = document.getElementById('ensemble-csv-file');
    const hasSelectedFile = selectedFileItem || (fileInput.files && fileInput.files.length > 0 && fileInput.files[0].size > 0);
    
    if (!hasSelectedFile) {
        isValid = false;
        errors.push('Please select a CSV file from the available files list or upload a new file');
        highlightFieldError('ensemble-csv-file', 'Please select a CSV file');
        highlightFieldError('ensemble-available-csv-files', 'Please select a CSV file');
    } else {
        clearFieldError('ensemble-csv-file');
        clearFieldError('ensemble-available-csv-files');
    }
    
    // Check weights for weighted ensemble
    if (ensembleType === 'weighted') {
        const weightStatus = document.getElementById('weight-status');
        if (weightStatus && !weightStatus.classList.contains('valid')) {
            isValid = false;
            errors.push('Please ensure weights sum to 1.0');
            highlightFieldError('weight-config', 'Please ensure weights sum to 1.0');
        } else {
            clearFieldError('weight-config');
        }
    }
    
    // Update submit button
    const submitBtn = document.getElementById('start-ensemble-training');
    if (submitBtn) {
        submitBtn.disabled = !isValid;
        submitBtn.textContent = isValid ? '🚀 Start Ensemble Training' : '❌ Fix Errors First';
        
        // Add visual feedback to button
        if (isValid) {
            submitBtn.classList.remove('btn-error');
            submitBtn.classList.add('btn-success');
        } else {
            submitBtn.classList.remove('btn-success');
            submitBtn.classList.add('btn-error');
        }
    }
    
    // Show error summary if there are errors
    showValidationErrorSummary(errors);
    
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
    
    // Check for data file - either from available files list or file input
    let csvPath = '';
    let csvFile = null;
    
    // First, check if a file is selected from the available files list
    const selectedFileItem = document.querySelector('#ensemble-available-csv-files .csv-file-item.selected');
    if (selectedFileItem) {
        csvPath = selectedFileItem.dataset.path;
    } 
    // If no file selected from list, check if a file was uploaded via file input
    else {
        const fileInput = document.getElementById('ensemble-csv-file');
        if (fileInput.files && fileInput.files.length > 0 && fileInput.files[0].size > 0) {
            csvFile = fileInput.files[0];
        }
    }
    
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
        csvPath,
        advancedOptions
    };
}

/**
 * Upload CSV file to server
 */
async function uploadCsvFile(file) {
    console.log('Uploading CSV file:', file.name);
    
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = async function(e) {
            try {
                const content = e.target.result.split(',')[1]; // Remove data:... prefix
                const fileData = {
                    name: file.name,
                    content: content
                };
                
                const result = await eel.upload_csv_file(fileData)();
                resolve(result);
            } catch (error) {
                reject(error);
            }
        };
        reader.onerror = () => reject(new Error('Failed to read file'));
        reader.readAsDataURL(file);
    });
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
        
        // Determine CSV file path
        let csvFilePath = '';
        
        if (formData.csvPath) {
            // Use existing file path
            csvFilePath = formData.csvPath;
            updateEnsembleTrainingStatus('Using selected CSV file...', 10);
        } else if (formData.csvFile) {
            // Upload new file
            updateEnsembleTrainingStatus('Uploading CSV file...', 10);
            const csvFileData = await uploadCsvFile(formData.csvFile);
            if (!csvFileData.success) {
                throw new Error('Failed to upload CSV file: ' + csvFileData.error);
            }
            csvFilePath = csvFileData.file_path;
        } else {
            throw new Error('No CSV file selected. Please select a file from the available files list or upload a new file.');
        }
        
        // Update status
        updateEnsembleTrainingStatus('Starting ensemble training...', 20);
        
        // Start training process
        const trainingId = await eel.start_ensemble_training(
            formData.ensembleType,
            formData.ensembleName,
            formData.selectedModels,
            csvFilePath,
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
                hideEnsembleTrainingProgress();
                loadEnsembleTrainingHistory();
            } else if (status.status === 'failed' || status.status === 'error') {
                console.error('Ensemble training failed:', status.message);
                showNotification('Ensemble training failed: ' + status.message, 'error');
                ensembleTrainingInProgress = false;
                hideEnsembleTrainingProgress();
            } else {
                // Continue monitoring
                setTimeout(() => monitorEnsembleTrainingProgress(trainingId), 2000);
            }
        }
    } catch (error) {
        console.error('Error monitoring ensemble training:', error);
        ensembleTrainingInProgress = false;
        hideEnsembleTrainingProgress();
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
    
    // Clear CSV file selection
    document.querySelectorAll('#ensemble-available-csv-files .csv-file-item').forEach(item => {
        item.classList.remove('selected');
    });
    
    // Reset file input display
    const fileDisplay = document.querySelector('#ensemble-training-content .file-input-display');
    const fileText = document.querySelector('#ensemble-training-content .file-input-text');
    const fileIcon = document.querySelector('#ensemble-training-content .file-input-icon');
    if (fileDisplay && fileText && fileIcon) {
        fileDisplay.classList.remove('has-file');
        fileText.textContent = 'Choose CSV file or drag & drop';
        fileIcon.textContent = '📁';
    }
    
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
 * Load available CSV files
 */
async function loadAvailableCSVFiles() {
    try {
        const csvFiles = await eel.get_available_csv_files()();
        displayCSVFiles(csvFiles);
    } catch (error) {
        console.error('Error loading CSV files:', error);
        displayCSVFilesError();
    }
}

/**
 * Display CSV files in the UI
 */
function displayCSVFiles(csvFiles) {
    const container = document.getElementById('ensemble-available-csv-files');
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
            <div class="csv-file-size">${formatFileSize(file.size)}</div>
        </div>
    `).join('');

    // Add click handlers for CSV file selection
    container.querySelectorAll('.csv-file-item').forEach(item => {
        item.addEventListener('click', () => {
            selectCSVFile(item);
        });
    });
}

/**
 * Display CSV files error
 */
function displayCSVFilesError() {
    const container = document.getElementById('ensemble-available-csv-files');
    if (container) {
        container.innerHTML = '<div class="error">Error loading CSV files. Please check the console for details.</div>';
    }
}

/**
 * Select CSV file from available files list
 */
function selectCSVFile(item) {
    // Remove previous selection
    document.querySelectorAll('#ensemble-available-csv-files .csv-file-item').forEach(i => {
        i.classList.remove('selected');
    });

    // Add selection to clicked item
    item.classList.add('selected');

    // Update file input
    const filePath = item.dataset.path;
    const fileInput = document.getElementById('ensemble-csv-file');
    if (fileInput) {
        // Create a fake file object for the selected CSV
        const fileName = filePath.split('/').pop();
        const fakeFile = new File([''], fileName, { type: 'text/csv' });
        const dataTransfer = new DataTransfer();
        dataTransfer.items.add(fakeFile);
        fileInput.files = dataTransfer.files;
        
        // Update display
        handleCsvFileChange({ target: fileInput });
        
        // Trigger validation
        validateEnsembleForm();
    }
}

/**
 * Format file size
 */
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

/**
 * Convert file to base64
 */
function fileToBase64(file) {
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

/**
 * Clear all validation errors
 */
function clearValidationErrors() {
    // Remove error classes from all form elements
    const errorElements = document.querySelectorAll('.field-error, .error-indicator');
    errorElements.forEach(el => el.remove());
    
    // Clear error classes from form groups
    const formGroups = document.querySelectorAll('.form-group');
    formGroups.forEach(group => {
        group.classList.remove('has-error');
        const errorMsg = group.querySelector('.error-message');
        if (errorMsg) errorMsg.remove();
    });
}

/**
 * Highlight field error
 */
function highlightFieldError(fieldId, message) {
    const field = document.getElementById(fieldId);
    if (!field) return;
    
    // Find the form group containing this field
    let formGroup = field.closest('.form-group');
    if (!formGroup) {
        // If no form group, create one or use the field itself
        formGroup = field;
    }
    
    // Add error class to form group
    formGroup.classList.add('has-error');
    
    // Add error indicator
    const errorIndicator = document.createElement('div');
    errorIndicator.className = 'error-indicator';
    errorIndicator.innerHTML = `⚠️ ${message}`;
    
    // Insert error indicator after the field
    if (field.nextSibling) {
        formGroup.insertBefore(errorIndicator, field.nextSibling);
    } else {
        formGroup.appendChild(errorIndicator);
    }
}

/**
 * Clear field error
 */
function clearFieldError(fieldId) {
    const field = document.getElementById(fieldId);
    if (!field) return;
    
    // Find the form group containing this field
    let formGroup = field.closest('.form-group');
    if (!formGroup) {
        formGroup = field;
    }
    
    // Remove error class
    formGroup.classList.remove('has-error');
    
    // Remove error indicator
    const errorIndicator = formGroup.querySelector('.error-indicator');
    if (errorIndicator) {
        errorIndicator.remove();
    }
}

/**
 * Show validation error summary
 */
function showValidationErrorSummary(errors) {
    // Remove existing error summary
    const existingSummary = document.getElementById('validation-error-summary');
    if (existingSummary) {
        existingSummary.remove();
    }
    
    if (errors.length === 0) {
        return;
    }
    
    // Create error summary
    const errorSummary = document.createElement('div');
    errorSummary.id = 'validation-error-summary';
    errorSummary.className = 'validation-error-summary';
    
    const errorList = errors.map(error => `<li>${error}</li>`).join('');
    errorSummary.innerHTML = `
        <div class="error-summary-header">
            <span class="error-icon">⚠️</span>
            <strong>Please fix the following errors:</strong>
        </div>
        <ul class="error-list">${errorList}</ul>
    `;
    
    // Insert error summary at the top of the form
    const form = document.getElementById('ensemble-training-form');
    if (form) {
        form.insertBefore(errorSummary, form.firstChild);
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