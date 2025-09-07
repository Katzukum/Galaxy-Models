# Galaxy Models Dashboard - Project Structure

## Overview
This is a modular, Agile-style project structure for the Galaxy Models Dashboard EEL application.

## Project Structure
```
web/
├── index.html                 # Main HTML file with tab navigation
├── css/
│   └── main.css              # Main stylesheet with theme variables
├── js/
│   ├── main.js               # Main application logic and tab management
│   ├── model-library.js      # Model Library tab functionality
│   └── training.js           # Training tab functionality
└── tabs/
    ├── model-library/
    │   ├── model-library.html    # Model Library tab HTML
    │   └── model-library.css     # Model Library specific styles
    └── training/
        ├── training.html         # Training tab HTML
        └── training.css          # Training specific styles
```

## Features

### 🎨 **Theme System**
- **CSS Custom Properties**: Master accent color variable (`--accent-color`) for easy theme changes
- **Dark Glass Theme**: Modern glassmorphism design with red accent color
- **Responsive Design**: Mobile-friendly layout

### 📱 **Tab System**
- **Model Library**: Current functionality for displaying YAML model configurations
- **Training**: Complete training interface with parameter configuration for all model types

### 🔧 **Development Benefits**
- **Modular Structure**: Each tab has its own folder and files
- **Separation of Concerns**: HTML, CSS, and JS are properly separated
- **Easy Maintenance**: Add new tabs by creating new folders and files
- **Scalable**: Structure supports team development and Agile workflows

## Adding New Tabs

1. Create a new folder in `web/tabs/your-tab-name/`
2. Add HTML file: `web/tabs/your-tab-name/your-tab-name.html`
3. Add CSS file: `web/tabs/your-tab-name/your-tab-name.css` (optional)
4. Add JS file: `web/js/your-tab-name.js`
5. Update `web/index.html` to include the new tab button and content
6. Update `web/js/main.js` to handle the new tab

## Theme Customization

To change the accent color, modify the CSS variable in `web/css/main.css`:

```css
:root {
    --accent-color: #00ff00; /* Change to your preferred color */
}
```

This will update the entire theme automatically.
