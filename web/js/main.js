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
        console.log(`[MAIN] Switching to tab: ${tabName}`);
        
        // Update active tab button
        document.querySelectorAll('.tab-button').forEach(btn => {
            btn.classList.remove('active');
        });
        const tabButton = document.querySelector(`[data-tab="${tabName}"]`);
        if (tabButton) {
            tabButton.classList.add('active');
            console.log(`[MAIN] Tab button activated: ${tabName}`);
        } else {
            console.error(`[MAIN] Tab button not found: ${tabName}`);
        }

        // Update active tab content
        document.querySelectorAll('.tab-content').forEach(content => {
            content.classList.remove('active');
        });
        const tabContent = document.querySelector(`#${tabName}-content`);
        if (tabContent) {
            tabContent.classList.add('active');
            console.log(`[MAIN] Tab content activated: ${tabName}`);
        } else {
            console.error(`[MAIN] Tab content not found: ${tabName}`);
        }

        this.currentTab = tabName;

        // Load tab-specific content
        console.log(`[MAIN] Loading content for tab: ${tabName}`);
        this.loadTabContent(tabName);
    }

    loadInitialTab() {
        this.loadTabContent(this.currentTab);
    }

    loadTabContent(tabName) {
        console.log(`[MAIN] loadTabContent called for: ${tabName}`);
        
        switch(tabName) {
            case 'model-library':
                console.log('[MAIN] Loading model-library content');
                if (typeof ModelLibrary !== 'undefined') {
                    console.log('[MAIN] ModelLibrary found, calling loadModels()');
                    ModelLibrary.loadModels();
                } else {
                    console.error('[MAIN] ModelLibrary not found');
                }
                break;
            case 'training':
                console.log('[MAIN] Loading training content');
                if (typeof Training !== 'undefined') {
                    console.log('[MAIN] Training found, calling loadTraining()');
                    Training.loadTraining();
                } else {
                    console.error('[MAIN] Training not found');
                }
                break;
            case 'ensemble-training':
                console.log('[MAIN] Loading ensemble-training content');
                console.log('[MAIN] Checking for initializeEnsembleTraining function...');
                console.log('[MAIN] typeof initializeEnsembleTraining:', typeof initializeEnsembleTraining);
                console.log('[MAIN] window.initializeEnsembleTraining:', typeof window.initializeEnsembleTraining);
                
                // Try multiple ways to find the function
                let initFunction = null;
                if (typeof initializeEnsembleTraining === 'function') {
                    initFunction = initializeEnsembleTraining;
                    console.log('[MAIN] Found initializeEnsembleTraining as global function');
                } else if (typeof window.initializeEnsembleTraining === 'function') {
                    initFunction = window.initializeEnsembleTraining;
                    console.log('[MAIN] Found initializeEnsembleTraining on window object');
                } else {
                    console.error('[MAIN] initializeEnsembleTraining function not found anywhere');
                    console.log('[MAIN] Available global functions:', Object.keys(window).filter(key => typeof window[key] === 'function'));
                }
                
                if (initFunction) {
                    console.log('[MAIN] Calling initializeEnsembleTraining...');
                    try {
                        initFunction();
                        console.log('[MAIN] initializeEnsembleTraining called successfully');
                    } catch (error) {
                        console.error('[MAIN] Error calling initializeEnsembleTraining:', error);
                    }
                }
                break;
            case 'backtesting':
                console.log('[MAIN] Loading backtesting content');
                if (window.BacktestingTab) {
                    console.log('[MAIN] BacktestingTab found, calling loadContent()');
                    window.BacktestingTab.loadContent();
                } else {
                    console.error('[MAIN] BacktestingTab not found');
                }
                break;
            case 'api-hosting':
                console.log('[MAIN] Loading api-hosting content');
                if (window.APIHostingTab) {
                    console.log('[MAIN] APIHostingTab found, calling loadContent()');
                    window.APIHostingTab.loadContent();
                } else {
                    console.error('[MAIN] APIHostingTab not found');
                }
                break;
            default:
                console.log(`[MAIN] Unknown tab: ${tabName}`);
        }
    }
}

// Initialize the app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.galaxyApp = new GalaxyModelsApp();
});
