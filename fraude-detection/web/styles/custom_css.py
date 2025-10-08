"""
Estilos CSS personalizados para la aplicación de Detección de Fraude
"""

CUSTOM_CSS = """
    <style>
    /* Importar fuente moderna */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    /* Estilos generales */
    * {
        font-family: 'Inter', sans-serif;
    }

    /* Header principal con gradiente */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    }

    .main-header h1 {
        font-size: 3rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }

    .main-header p {
        font-size: 1.2rem;
        margin-top: 0.5rem;
        opacity: 0.9;
    }

    /* Cards personalizadas */
    .metric-card {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
        margin: 0.5rem 0;
        transition: transform 0.3s ease;
    }

    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 24px rgba(0,0,0,0.3);
    }

    .metric-card h3 {
        color: #a8dadc;
        font-size: 0.9rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .metric-card .value {
        color: white;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
    }

    .metric-card .subtitle {
        color: #e0e0e0;
        font-size: 0.85rem;
        margin-top: 0.5rem;
    }

    /* Botones personalizados */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }

    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }

    /* Sidebar personalizada */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }

    /* Forzar color blanco a TODOS los elementos de texto en sidebar */
    [data-testid="stSidebar"] * {
        color: white !important;
    }

    /* Todos los labels */
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] label *,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] label * {
        color: white !important;
    }

    /* Radio buttons */
    [data-testid="stSidbar"] .stRadio > label,
    [data-testid="stSidebar"] .stRadio label,
    [data-testid="stSidebar"] div[role="radiogroup"] label,
    [data-testid="stSidebar"] div[role="radiogroup"] label p,
    [data-testid="stSidebar"] div[role="radiogroup"] label span,
    [data-testid="stSidebar"] div[data-baseweb="radio"] label {
        color: white !important;
        font-size: 1.1rem !important;
        font-weight: 600 !important;
    }

    /* Paragraphs y spans en sidebar */
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] div {
        color: white !important;
    }

    /* Markdown en sidebar */
    [data-testid="stSidebar"] .stMarkdown,
    [data-testid="stSidebar"] .stMarkdown * {
        color: white !important;
    }

    /* Metrics en sidebar */
    [data-testid="stSidebar"] [data-testid="stMetricLabel"],
    [data-testid="stSidebar"] [data-testid="stMetricValue"],
    [data-testid="stSidebar"] .stMetric label,
    [data-testid="stSidebar"] .stMetric {
        color: white !important;
    }

    /* Selectbox */
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stSelectbox {
        color: white !important;
    }

    /* Alertas personalizadas */
    .alert-fraud {
        background: linear-gradient(135deg, #ff6b6b 0%, #c92a2a 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: 700;
        box-shadow: 0 8px 20px rgba(255, 107, 107, 0.4);
        animation: pulse 2s infinite;
    }

    .alert-legit {
        background: linear-gradient(135deg, #51cf66 0%, #2b8a3e 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: 700;
        box-shadow: 0 8px 20px rgba(81, 207, 102, 0.4);
    }

    @keyframes pulse {
        0%, 100% {
            transform: scale(1);
        }
        50% {
            transform: scale(1.02);
        }
    }

    /* Tabs personalizadas */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: #2d3748;
        border-radius: 10px;
        padding: 10px 20px;
        color: white;
        font-weight: 600;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }

    /* Info boxes */
    .info-box {
        background: rgba(102, 126, 234, 0.1);
        border-left: 4px solid #667eea;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }

    .success-box {
        background: rgba(81, 207, 102, 0.1);
        border-left: 4px solid #51cf66;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }

    .warning-box {
        background: rgba(255, 107, 107, 0.1);
        border-left: 4px solid #ff6b6b;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }

    /* Ocultar elementos de Streamlit */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Scrollbar personalizada */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }

    ::-webkit-scrollbar-track {
        background: #1a1a2e;
    }

    ::-webkit-scrollbar-thumb {
        background: #667eea;
        border-radius: 5px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: #764ba2;
    }
    </style>
"""

SIDEBAR_CSS = """
    <style>
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    [data-testid="stSidebar"] .stMarkdown {
        color: white !important;
    }
    </style>
"""
