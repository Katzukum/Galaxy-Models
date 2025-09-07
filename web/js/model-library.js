// Model Library Tab JavaScript
class ModelLibrary {
    static async loadModels() {
        const container = document.getElementById('models-container');
        const stats = document.getElementById('model-library-stats');
        
        try {
            container.innerHTML = '<div class="loading">Loading models...</div>';
            stats.innerHTML = '<h3>Loading Models...</h3><p>Scanning the Models folder for YAML configurations</p>';
            
            const models = await eel.get_models()();
            
            if (models.length === 0) {
                container.innerHTML = '<div class="error">No YAML configuration files found in the Models folder.</div>';
                stats.innerHTML = '<h3>No Models Found</h3><p>No YAML configuration files were found in the Models folder</p>';
                return;
            }
            
            // Update stats
            stats.innerHTML = `<h3>📊 Model Statistics</h3><p>Found ${models.length} model${models.length !== 1 ? 's' : ''} ready for deployment</p>`;
            
            // Create model cards
            const modelsGrid = document.createElement('div');
            modelsGrid.className = 'models-grid';
            
            models.forEach(model => {
                const card = document.createElement('div');
                card.className = 'model-card';
                card.onclick = () => ModelLibrary.showModelDetails(model);
                
                card.innerHTML = `
                    <div class="model-header">
                        <div class="model-name">${model.name}</div>
                        <div class="model-type">${model.type}</div>
                    </div>
                    <div class="model-path">${model.config_path}</div>
                `;
                
                modelsGrid.appendChild(card);
            });
            
            container.innerHTML = '';
            container.appendChild(modelsGrid);
            
        } catch (error) {
            console.error('Error loading models:', error);
            container.innerHTML = `<div class="error">Error loading models: ${error.message}</div>`;
            stats.innerHTML = '<h3>❌ Error</h3><p>Failed to load models from the Models folder</p>';
        }
    }
    
    static async showModelDetails(model) {
        try {
            const details = await eel.get_model_details(model.config_path)();
            
            if (details.error) {
                alert(`Error loading model details: ${details.error}`);
                return;
            }
            
            // Create a detailed view (you can expand this)
            const detailsText = JSON.stringify(details, null, 2);
            alert(`Model Details for ${model.name}:\n\n${detailsText}`);
            
        } catch (error) {
            console.error('Error loading model details:', error);
            alert(`Error loading model details: ${error.message}`);
        }
    }
}
