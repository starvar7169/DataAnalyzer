# 📊 DataGPT - AI-Powered Data Analyst

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Active-success.svg)

**An intelligent no-code platform that democratizes data analysis through AI-powered preprocessing and insights.**

[Features](#-features) • [Installation](#-installation) • [Usage](#-usage) • [Architecture](#-architecture) • [Contributing](#-contributing)

</div>

---

## 🎯 Overview

DataGPT is an AI-powered data analysis tool that combines traditional data preprocessing capabilities with advanced Large Language Model (LLM) integration. It provides data enthusiasts and professionals with an intuitive interface to clean, analyze, and derive insights from their datasets without writing a single line of code.

### ✨ Key Highlights

- 🤖 **AI-Powered Analysis** - Leverages Groq LLM for intelligent data insights
- 📈 **Comprehensive Preprocessing** - Handles missing values, outliers, normalization, and encoding
- 📊 **Interactive Visualizations** - Real-time charts and correlation matrices
- 💬 **Conversational Interface** - Ask questions about your data in natural language
- 🔄 **Multi-Format Support** - CSV, JSON, Excel, and TXT files
- ⚡ **Fast Processing** - Optimized for large datasets

---

## 🚀 Features

### Data Processing
- **Automatic Type Detection** - Smart identification of numeric and categorical columns
- **Missing Value Handling** - Multiple strategies (mean, median, mode, drop)
- **Outlier Detection & Removal** - Statistical methods with visualizations
- **Data Normalization** - Min-Max and Standard scaling
- **One-Hot Encoding** - Automatic categorical variable transformation

### AI Capabilities
- **Natural Language Queries** - Ask questions about your data conversationally
- **Intelligent Insights** - AI-generated analysis and recommendations
- **Pattern Recognition** - Automated correlation and trend detection
- **Report Generation** - Export-ready analysis summaries

### Visualization
- **Interactive Scatter Plots** - Explore relationships between variables
- **Correlation Heatmaps** - Identify strong correlations
- **Box Plots** - Visualize distributions and outliers
- **Custom Charts** - Flexible plotting with Plotly

---

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Groq API key ([Get one here](https://console.groq.com))

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/starvar7169/DataAnalyzer.git
   cd DataAnalyzer
   ```

2. **Create virtual environment**
   ```bash
   python -m venv .venv
   
   # Activate on Windows
   .venv\Scripts\activate
   
   # Activate on macOS/Linux
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**
   ```bash
   # Copy the example file
   cp .env.example .env
   
   # Edit .env and add your Groq API key
   # GROQ_API_KEY=your_actual_api_key_here
   ```

5. **Run the application**
   ```bash
   streamlit run app.py
   ```

The application will open in your default browser at `http://localhost:8501`

---

## 📖 Usage

### Quick Start

1. **Upload Your Data**
   - Click "Browse files" in the sidebar
   - Select a CSV, JSON, Excel, or TXT file
   - Preview your data instantly

2. **Choose Processing Options**
   - Select automatic processing or manual techniques
   - Configure missing value handling
   - Apply normalization or encoding as needed

3. **Analyze with AI**
   - Use the chat interface to ask questions
   - Get instant insights and recommendations
   - Generate visualizations on demand

4. **Export Results**
   - Download processed training/testing datasets
   - Save visualizations and reports
   - Share insights with your team


## 🏗️ Architecture

### Tech Stack

- **Frontend**: Streamlit
- **Data Processing**: Pandas, NumPy, Scikit-learn
- **Visualization**: Plotly, Matplotlib, Seaborn
- **AI/LLM**: Groq API (Claude/GPT models)
- **Backend**: Python 3.8+

### Project Structure

```
DataAnalyzer/
├── app.py                    # Main application entry point
├── src/
│   ├── components/           # UI components
│   ├── core/                 # Business logic
│   └── agents/               # AI agents
├── requirements.txt          # Dependencies
└── .env.example             # Environment template
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Anoushka Vats**
- GitHub: [@starvar7169](https://github.com/starvar7169)
- LinkedIn: [Connect with me](https://linkedin.com/in/your-profile)
- Email: anoushkavats71@gmail.com

---

## 🙏 Acknowledgments

- Built with ❤️ using Streamlit and Groq
- Inspired by the need for democratized data analysis
- Thanks to the open-source community

---

## 📊 Project Stats

![GitHub stars](https://img.shields.io/github/stars/starvar7169/DataAnalyzer?style=social)
![GitHub forks](https://img.shields.io/github/forks/starvar7169/DataAnalyzer?style=social)
![GitHub watchers](https://img.shields.io/github/watchers/starvar7169/DataAnalyzer?style=social)

---

<div align="center">

**If you find this project helpful, please consider giving it a ⭐!**

Made with ❤️ by [Anoushka Vats](https://github.com/starvar7169)

</div>
