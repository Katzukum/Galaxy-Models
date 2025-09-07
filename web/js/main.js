// Main application JavaScript
class GalaxyModelsApp {
    constructor() {
        this.currentTab = 'model-library';
        this.init();
    }

    init() {
        this.setupTabNavigation();
        this.loadInitialTab();
    }

    setupTabNavigation() {
        const tabButtons = document.querySelectorAll('.tab-button');
        tabButtons.forEach(button => {
            button.addEventListener('click', (e) => {
                const tabName = e.target.dataset.tab;
                this.switchTab(tabName);
            });
        });
    }

    switchTab(tabName) {
        // Update active tab button
        document.querySelectorAll('.tab-button').forEach(btn => {
            btn.classList.remove('active');
        });
        document.querySelector(`[data-tab="${tabName}"]`).classList.add('active');

        // Update active tab content
        document.querySelectorAll('.tab-content').forEach(content => {
            content.classList.remove('active');
        });
        document.querySelector(`#${tabName}-content`).classList.add('active');

        this.currentTab = tabName;

        // Load tab-specific content
        this.loadTabContent(tabName);
    }

    loadInitialTab() {
        this.loadTabContent(this.currentTab);
    }

    loadTabContent(tabName) {
        switch(tabName) {
            case 'model-library':
                if (typeof ModelLibrary !== 'undefined') {
                    ModelLibrary.loadModels();
                }
                break;
            case 'training':
                if (typeof Training !== 'undefined') {
                    Training.loadTraining();
                }
                break;
            case 'backtesting':
                if (window.BacktestingTab) {
                    window.BacktestingTab.loadContent();
                }
                break;
            case 'api-hosting':
                if (window.APIHostingTab) {
                    window.APIHostingTab.loadContent();
                }
                break;
        }
    }
}

// Initialize the app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.galaxyApp = new GalaxyModelsApp();
});
