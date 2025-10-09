"""
DataGPT - AI-Powered Data Analyst
Split-screen interface: Preprocessing (Left) + AI Analysis (Right)

Author: Anoushka Vats
GitHub: starvar7169
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import io

# Page configuration
st.set_page_config(
    page_title="DataGPT - AI Data Analyst",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for beautiful split-screen layout
st.markdown("""
<style>
    /* Main container */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Remove default padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 0rem;
        max-width: 100%;
    }
    
    /* Header styling */
    .main-header {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        margin-bottom: 2rem;
        text-align: center;
    }
    
    .main-header h1 {
        color: #667eea;
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
    }
    
    .main-header p {
        color: #666;
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
    }
    
    /* Split panel styling */
    .split-panel {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        height: calc(100vh - 200px);
        overflow-y: auto;
        margin-bottom: 2rem;
    }
    
    .panel-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #333;
        margin-bottom: 1.5rem;
        padding-bottom: 1rem;
        border-bottom: 3px solid #667eea;
    }
    
    .left-panel {
        border-right: 2px solid #f0f0f0;
    }
    
    /* Button styling */
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        padding: 0.6rem 1.2rem;
        font-weight: 600;
        border: none;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* File uploader */
    .stFileUploader {
        border: 2px dashed #667eea;
        border-radius: 10px;
        padding: 1.5rem;
        background: #f8f9ff;
    }
    
    /* Selectbox styling */
    .stSelectbox {
        margin-bottom: 1rem;
    }
    
    /* Checkbox styling */
    .stCheckbox {
        margin: 0.5rem 0;
    }
    
    /* Dataframe styling */
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: #f8f9ff;
        border-radius: 8px;
        font-weight: 600;
    }
    
    /* Metric cards */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        color: #667eea;
    }
    
    /* Chat message styling */
    .chat-message {
        background: #f8f9ff;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #667eea;
    }
    
    .user-message {
        background: #e8f0fe;
        border-left: 4px solid #4285f4;
    }
    
    .ai-message {
        background: #f3e8ff;
        border-left: 4px solid #9333ea;
    }
    
    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #667eea;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #764ba2;
    }
    
    /* Success/Error messages */
    .stSuccess {
        background: #d4edda;
        border-left: 4px solid #28a745;
        border-radius: 8px;
    }
    
    .stError {
        background: #f8d7da;
        border-left: 4px solid #dc3545;
        border-radius: 8px;
    }
    
    /* Info messages */
    .stInfo {
        background: #d1ecf1;
        border-left: 4px solid #17a2b8;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize session state variables"""
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'processed_df' not in st.session_state:
        st.session_state.processed_df = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = []


def render_header():
    """Render application header"""
    st.markdown("""
    <div class="main-header">
        <h1>📊 DataGPT</h1>
        <p>AI-Powered Data Analysis & Preprocessing Platform</p>
    </div>
    """, unsafe_allow_html=True)


def preprocessing_panel():
    """Left panel: Data preprocessing and manipulation"""
    st.markdown('<div class="split-panel left-panel">', unsafe_allow_html=True)
    st.markdown('<h2 class="panel-header">🔧 Data Preprocessing</h2>', unsafe_allow_html=True)
    
    # File Upload Section
    st.subheader("📁 Upload Dataset")
    uploaded_file = st.file_uploader(
        "Choose a file", 
        type=["csv", "json", "xlsx", "txt"],
        help="Upload your dataset to begin analysis"
    )
    
    if uploaded_file is not None:
        try:
            # Read file based on extension
            file_extension = uploaded_file.name.split('.')[-1]
            
            if file_extension == 'csv':
                df = pd.read_csv(uploaded_file)
            elif file_extension == 'json':
                df = pd.read_json(uploaded_file)
            elif file_extension == 'xlsx':
                df = pd.read_excel(uploaded_file)
            elif file_extension == 'txt':
                df = pd.read_csv(uploaded_file, delimiter='\t')
            
            st.session_state.df = df.copy()
            st.session_state.processed_df = df.copy()
            
            st.success(f"✅ Successfully loaded {uploaded_file.name}")
            
            # Dataset Overview
            with st.expander("📊 Dataset Overview", expanded=True):
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Rows", df.shape[0])
                with col2:
                    st.metric("Columns", df.shape[1])
                with col3:
                    st.metric("Missing", df.isnull().sum().sum())
                with col4:
                    st.metric("Duplicates", df.duplicated().sum())
                
                st.dataframe(df.head(10), use_container_width=True)
            
            # Data Information
            with st.expander("ℹ️ Data Information"):
                info_choice = st.selectbox("Select Info Type:", 
                                          ["Data Types", "Statistical Summary", "Column Names", "Memory Usage"])
                
                if info_choice == "Data Types":
                    st.write(df.dtypes.to_frame(name='Data Type'))
                elif info_choice == "Statistical Summary":
                    st.write(df.describe())
                elif info_choice == "Column Names":
                    st.write(pd.DataFrame(df.columns, columns=['Columns']))
                elif info_choice == "Memory Usage":
                    buffer = io.StringIO()
                    df.info(buf=buffer)
                    st.text(buffer.getvalue())
            
            st.divider()
            
            # Preprocessing Options
            st.subheader("⚙️ Preprocessing Options")
            
            # Column Type Processing
            with st.expander("🔄 Data Type Processing"):
                process_option = st.selectbox(
                    "Choose Processing Method", 
                    ["Keep All", "Drop Non-Numeric", "One-Hot Encoding"]
                )
                
                if st.button("Apply Processing", key="process_btn"):
                    if process_option == "Drop Non-Numeric":
                        st.session_state.processed_df = st.session_state.processed_df.select_dtypes(include=[np.number])
                        st.success("✅ Non-numeric columns dropped!")
                    elif process_option == "One-Hot Encoding":
                        st.session_state.processed_df = pd.get_dummies(
                            st.session_state.processed_df, 
                            drop_first=True
                        )
                        st.success("✅ One-hot encoding applied!")
            
            # Missing Values
            with st.expander("🔍 Handle Missing Values"):
                missing_action = st.radio(
                    "Choose Action:",
                    ["None", "Drop Rows", "Drop Columns", "Fill with Strategy"]
                )
                
                if missing_action == "Drop Rows":
                    if st.button("Drop Rows with Missing Values"):
                        st.session_state.processed_df = st.session_state.processed_df.dropna()
                        st.success(f"✅ Dropped rows. New shape: {st.session_state.processed_df.shape}")
                
                elif missing_action == "Drop Columns":
                    if st.button("Drop Columns with All Missing"):
                        st.session_state.processed_df = st.session_state.processed_df.dropna(axis=1, how='all')
                        st.success(f"✅ Dropped columns. New shape: {st.session_state.processed_df.shape}")
                
                elif missing_action == "Fill with Strategy":
                    fill_method = st.selectbox("Filling Method:", ["Mean", "Median", "Mode", "Forward Fill", "Backward Fill"])
                    
                    if st.button("Fill Missing Values"):
                        numeric_cols = st.session_state.processed_df.select_dtypes(include=[np.number]).columns
                        
                        if fill_method == "Mean":
                            st.session_state.processed_df[numeric_cols] = st.session_state.processed_df[numeric_cols].fillna(
                                st.session_state.processed_df[numeric_cols].mean()
                            )
                        elif fill_method == "Median":
                            st.session_state.processed_df[numeric_cols] = st.session_state.processed_df[numeric_cols].fillna(
                                st.session_state.processed_df[numeric_cols].median()
                            )
                        elif fill_method == "Mode":
                            st.session_state.processed_df[numeric_cols] = st.session_state.processed_df[numeric_cols].fillna(
                                st.session_state.processed_df[numeric_cols].mode().iloc[0]
                            )
                        elif fill_method == "Forward Fill":
                            st.session_state.processed_df = st.session_state.processed_df.fillna(method='ffill')
                        elif fill_method == "Backward Fill":
                            st.session_state.processed_df = st.session_state.processed_df.fillna(method='bfill')
                        
                        st.success(f"✅ Filled missing values using {fill_method}")
            
            # Outlier Detection
            with st.expander("📉 Outlier Detection & Removal"):
                if st.button("Detect & Remove Outliers"):
                    original_shape = st.session_state.processed_df.shape
                    numeric_cols = st.session_state.processed_df.select_dtypes(include=[np.number]).columns
                    
                    for col in numeric_cols:
                        Q1 = st.session_state.processed_df[col].quantile(0.25)
                        Q3 = st.session_state.processed_df[col].quantile(0.75)
                        IQR = Q3 - Q1
                        
                        lower_bound = Q1 - 1.5 * IQR
                        upper_bound = Q3 + 1.5 * IQR
                        
                        st.session_state.processed_df = st.session_state.processed_df[
                            (st.session_state.processed_df[col] >= lower_bound) & 
                            (st.session_state.processed_df[col] <= upper_bound)
                        ]
                    
                    st.success(f"✅ Outliers removed! Shape: {original_shape} → {st.session_state.processed_df.shape}")
            
            # Data Scaling
            with st.expander("📏 Data Scaling"):
                scale_method = st.selectbox("Scaling Method:", ["None", "Min-Max Scaler", "Standard Scaler"])
                
                if scale_method != "None" and st.button("Apply Scaling"):
                    numeric_cols = st.session_state.processed_df.select_dtypes(include=[np.number]).columns
                    
                    if scale_method == "Min-Max Scaler":
                        scaler = MinMaxScaler()
                        st.session_state.processed_df[numeric_cols] = scaler.fit_transform(
                            st.session_state.processed_df[numeric_cols]
                        )
                        st.success("✅ Min-Max scaling applied!")
                    
                    elif scale_method == "Standard Scaler":
                        scaler = StandardScaler()
                        st.session_state.processed_df[numeric_cols] = scaler.fit_transform(
                            st.session_state.processed_df[numeric_cols]
                        )
                        st.success("✅ Standard scaling applied!")
            
            st.divider()
            
            # Train-Test Split
            st.subheader("✂️ Train-Test Split")
            test_size = st.slider("Test Size", 0.1, 0.5, 0.2, 0.05)
            
            if st.button("Split & Download Data"):
                X = st.session_state.processed_df.drop(columns=st.session_state.processed_df.columns[-1])
                y = st.session_state.processed_df[st.session_state.processed_df.columns[-1]]
                
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42
                )
                
                train_data = pd.concat([X_train, y_train], axis=1)
                test_data = pd.concat([X_test, y_test], axis=1)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.download_button(
                        label="⬇️ Download Training Data",
                        data=train_data.to_csv(index=False),
                        file_name='train_data.csv',
                        mime='text/csv',
                        use_container_width=True
                    )
                
                with col2:
                    st.download_button(
                        label="⬇️ Download Testing Data",
                        data=test_data.to_csv(index=False),
                        file_name='test_data.csv',
                        mime='text/csv',
                        use_container_width=True
                    )
                
                st.success(f"✅ Split complete! Train: {len(train_data)} rows, Test: {len(test_data)} rows")
            
            # Download Processed Data
            st.download_button(
                label="💾 Download Processed Dataset",
                data=st.session_state.processed_df.to_csv(index=False),
                file_name='processed_data.csv',
                mime='text/csv',
                use_container_width=True
            )
        
        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")
    
    else:
        st.info("👆 Upload a dataset to get started!")
    
    st.markdown('</div>', unsafe_allow_html=True)


def analysis_panel():
    """Right panel: AI-powered analysis and visualization"""
    st.markdown('<div class="split-panel">', unsafe_allow_html=True)
    st.markdown('<h2 class="panel-header">🤖 AI Analysis & Insights</h2>', unsafe_allow_html=True)
    
    if st.session_state.processed_df is not None:
        df = st.session_state.processed_df
        
        # Quick Stats
        st.subheader("📈 Quick Statistics")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Rows", df.shape[0], delta=None)
        with col2:
            st.metric("Total Columns", df.shape[1], delta=None)
        with col3:
            st.metric("Numeric Columns", len(df.select_dtypes(include=[np.number]).columns))
        
        st.divider()
        
        # Correlation Analysis
        st.subheader("🔗 Correlation Analysis")
        numeric_df = df.select_dtypes(include=[np.number])
        
        if not numeric_df.empty:
            with st.expander("View Correlation Matrix", expanded=True):
                corr_matrix = numeric_df.corr()
                
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", center=0, ax=ax, 
                           square=True, linewidths=1, cbar_kws={"shrink": 0.8})
                ax.set_title("Correlation Heatmap", fontsize=16, fontweight='bold')
                st.pyplot(fig)
                plt.close()
                
                # Strong correlations
                st.subheader("💪 Strong Correlations")
                strong_corr = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        if abs(corr_matrix.iloc[i, j]) > 0.7:
                            strong_corr.append({
                                'Feature 1': corr_matrix.columns[i],
                                'Feature 2': corr_matrix.columns[j],
                                'Correlation': round(corr_matrix.iloc[i, j], 3)
                            })
                
                if strong_corr:
                    st.dataframe(pd.DataFrame(strong_corr), use_container_width=True)
                else:
                    st.info("No strong correlations (>0.7) found")
        
        st.divider()
        
        # Interactive Visualizations
        st.subheader("📊 Interactive Visualizations")
        
        viz_type = st.selectbox("Choose Visualization:", 
                               ["Scatter Plot", "Box Plot", "Histogram", "Line Chart"])
        
        if viz_type == "Scatter Plot":
            col1, col2 = st.columns(2)
            with col1:
                x_col = st.selectbox("X-Axis:", df.columns, key="scatter_x")
            with col2:
                y_col = st.selectbox("Y-Axis:", df.columns, key="scatter_y")
            
            if st.button("Generate Scatter Plot"):
                fig = px.scatter(df, x=x_col, y=y_col, title=f"{x_col} vs {y_col}",
                               color_discrete_sequence=['#667eea'])
                st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "Box Plot":
            col_select = st.selectbox("Select Column:", numeric_df.columns)
            if st.button("Generate Box Plot"):
                fig = px.box(df, y=col_select, title=f"Box Plot: {col_select}",
                           color_discrete_sequence=['#764ba2'])
                st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "Histogram":
            col_select = st.selectbox("Select Column:", numeric_df.columns, key="hist_col")
            bins = st.slider("Number of Bins:", 10, 100, 30)
            if st.button("Generate Histogram"):
                fig = px.histogram(df, x=col_select, nbins=bins, title=f"Distribution: {col_select}",
                                 color_discrete_sequence=['#667eea'])
                st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "Line Chart":
            y_col = st.selectbox("Y-Axis:", numeric_df.columns)
            if st.button("Generate Line Chart"):
                fig = px.line(df, y=y_col, title=f"Trend: {y_col}",
                            color_discrete_sequence=['#764ba2'])
                st.plotly_chart(fig, use_container_width=True)
        
        st.divider()
        
        # AI Chat Interface
        st.subheader("💬 Ask AI About Your Data")
        
        # Display chat history
        for message in st.session_state.chat_history:
            if message['role'] == 'user':
                st.markdown(f'<div class="chat-message user-message">👤 <strong>You:</strong> {message["content"]}</div>', 
                          unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-message ai-message">🤖 <strong>AI:</strong> {message["content"]}</div>', 
                          unsafe_allow_html=True)
        
        # Chat input
        user_question = st.text_input("Ask a question about your data:", 
                                     placeholder="e.g., What are the main insights from this dataset?")
        
        if st.button("🚀 Get AI Insights", use_container_width=True):
            if user_question:
                st.session_state.chat_history.append({
                    'role': 'user',
                    'content': user_question
                })
                
                # Simulated AI response (replace with actual Groq API call)
                ai_response = f"""Based on your dataset with {df.shape[0]} rows and {df.shape[1]} columns:
                
• The dataset contains {len(numeric_df.columns)} numeric features
• Missing values: {df.isnull().sum().sum()} total
• Data types are well-balanced for analysis

To get AI-powered insights, integrate the Groq API with your GROQ_API_KEY.
"""
                
                st.session_state.chat_history.append({
                    'role': 'assistant',
                    'content': ai_response
                })
                
                st.rerun()
            else:
                st.warning("⚠️ Please enter a question first!")
        
        # Clear chat
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()
    
    else:
        st.info("👈 Upload and preprocess data on the left panel to see analysis here!")
        
        # Show sample insights
        st.markdown("""
        ### 🌟 What you can do here:
        
        - 📊 **View correlation matrices** to find relationships between variables
        - 📈 **Generate interactive visualizations** (scatter plots, histograms, box plots)
        - 🤖 **Ask AI questions** about your data patterns and insights
        - 💡 **Get automated recommendations** for further analysis
        - 📋 **Export insights** and reports
        
        Upload your dataset to get started!
        """)
    
    st.markdown('</div>', unsafe_allow_html=True)


def main():
    """Main application logic"""
    initialize_session_state()
    
    # Render header
    render_header()
    
    # Create two-column layout
    left_col, right_col = st.columns([1, 1], gap="large")
    
    with left_col:
        preprocessing_panel()
    
    with right_col:
        analysis_panel()
    
    # Footer
    st.markdown("---")
    footer_col1, footer_col2, footer_col3 = st.columns(3)
    with footer_col1:
        st.caption("🚀 Built with Streamlit + Groq")
    with footer_col2:
        st.caption("⭐ [Star on GitHub](https://github.com/starvar7169/DataAnalyzer)")
    with footer_col3:
        st.caption("💼 Made with ❤️ by Anoushka Vats")


if __name__ == "__main__":
    main()
