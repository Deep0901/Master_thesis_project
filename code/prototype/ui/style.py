SWISS_GOV_CSS = """
<style>
/* Swiss Confederation Design System Colors */
:root {
    --swiss-red: #D8232A;
    --swiss-dark: #333333;
    --swiss-gray: #757575;
    --swiss-light-gray: #F5F5F5;
    --swiss-blue: #006699;
    --swiss-green: #4A9F35;
    --swiss-orange: #F29400;
    --swiss-white: #FFFFFF;
}

/* Main container styling */
.main .block-container {
    padding-top: 2rem;
    max-width: 1200px;
}

/* Swiss Header Banner */
.swiss-header {
    background: linear-gradient(90deg, var(--swiss-red) 0%, #B31B21 100%);
    color: white;
    padding: 20px 30px;
    margin: -1rem -1rem 2rem -1rem;
    border-radius: 0;
}

.swiss-header h1 {
    color: white !important;
    font-size: 1.8rem !important;
    margin: 0 !important;
    font-weight: 700 !important;
}

.swiss-header p {
    color: rgba(255,255,255,0.9);
    margin: 5px 0 0 0;
    font-size: 0.95rem;
}

/* Swiss Cross Logo */
.swiss-cross {
    display: inline-block;
    width: 32px;
    height: 32px;
    background: white;
    position: relative;
    margin-right: 15px;
    vertical-align: middle;
}

.swiss-cross:before {
    content: '';
    position: absolute;
    background: var(--swiss-red);
    top: 25%;
    left: 8%;
    width: 84%;
    height: 50%;
}

.swiss-cross:after {
    content: '';
    position: absolute;
    background: var(--swiss-red);
    top: 8%;
    left: 25%;
    width: 50%;
    height: 84%;
}

/* Search Box */
.search-container {
    background: white;
    border-radius: 8px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    padding: 25px;
    margin-bottom: 25px;
}

/* Result Cards */
.result-card {
    background: white;
    border: 1px solid #E0E0E0;
    border-radius: 8px;
    padding: 20px;
    margin-bottom: 15px;
    transition: box-shadow 0.2s ease;
    border-left: 4px solid var(--swiss-blue);
}

.result-card:hover {
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
}

.result-card.high-relevance {
    border-left-color: var(--swiss-green);
}

.result-card.medium-relevance {
    border-left-color: var(--swiss-orange);
}

.result-card.low-relevance {
    border-left-color: var(--swiss-red);
}

/* Score Badges */
.score-badge {
    display: inline-flex;
    align-items: center;
    padding: 6px 14px;
    border-radius: 20px;
    font-weight: 600;
    font-size: 0.85rem;
    margin-right: 8px;
}

.score-excellent {
    background: linear-gradient(135deg, #4A9F35, #3D8A2D);
    color: white;
}

.score-good {
    background: linear-gradient(135deg, #006699, #005580);
    color: white;
}

.score-moderate {
    background: linear-gradient(135deg, #F29400, #D98500);
    color: white;
}

.score-low {
    background: linear-gradient(135deg, #D8232A, #B81D23);
    color: white;
}

/* Factor Indicators */
.factor-bar {
    height: 8px;
    border-radius: 4px;
    background: #E0E0E0;
    margin: 5px 0;
    overflow: hidden;
}

.factor-fill {
    height: 100%;
    border-radius: 4px;
    transition: width 0.3s ease;
}

.factor-fill.recency { background: linear-gradient(90deg, #4A9F35, #6BC850); }
.factor-fill.completeness { background: linear-gradient(90deg, #006699, #0088CC); }
.factor-fill.resources { background: linear-gradient(90deg, #9B59B6, #B370CF); }
.factor-fill.similarity { background: linear-gradient(90deg, #F29400, #FFB340); }

/* Explanation Box */
.explanation-box {
    background: linear-gradient(135deg, #F8F9FA, #FFFFFF);
    border: 1px solid #E0E0E0;
    border-radius: 8px;
    padding: 15px;
    margin-top: 15px;
    font-size: 0.9rem;
}

.explanation-title {
    font-weight: 600;
    color: var(--swiss-dark);
    margin-bottom: 10px;
    display: flex;
    align-items: center;
}

.explanation-title svg {
    margin-right: 8px;
}

/* Tag Pills */
.tag-pill {
    display: inline-block;
    background: #E8F4F8;
    color: var(--swiss-blue);
    padding: 4px 10px;
    border-radius: 15px;
    font-size: 0.8rem;
    margin: 2px;
}

/* Organization Badge */
.org-badge {
    display: inline-flex;
    align-items: center;
    color: var(--swiss-gray);
    font-size: 0.85rem;
}

.org-badge svg {
    margin-right: 5px;
}

/* Format Badges */
.format-badge {
    display: inline-block;
    padding: 3px 8px;
    border-radius: 4px;
    font-size: 0.75rem;
    font-weight: 600;
    margin-right: 4px;
}

.format-csv { background: #E3F2FD; color: #1565C0; }
.format-json { background: #FFF3E0; color: #E65100; }
.format-xml { background: #F3E5F5; color: #7B1FA2; }
.format-pdf { background: #FFEBEE; color: #C62828; }
.format-api { background: #E8F5E9; color: #2E7D32; }
.format-geojson { background: #E0F2F1; color: #00695C; }
.format-xlsx { background: #E8EAF6; color: #303F9F; }
.format-default { background: #ECEFF1; color: #546E7A; }

/* Comparison Table */
.comparison-table {
    width: 100%;
    border-collapse: collapse;
    margin: 15px 0;
}

.comparison-table th {
    background: var(--swiss-light-gray);
    padding: 12px;
    text-align: left;
    font-weight: 600;
}

.comparison-table td {
    padding: 12px;
    border-bottom: 1px solid #E0E0E0;
}

/* Metrics Display */
.metric-card {
    background: white;
    border-radius: 8px;
    padding: 20px;
    text-align: center;
    border: 1px solid #E0E0E0;
}

.metric-value {
    font-size: 2rem;
    font-weight: 700;
    color: var(--swiss-blue);
}

.metric-label {
    color: var(--swiss-gray);
    font-size: 0.85rem;
    margin-top: 5px;
}

/* Sidebar Styling */
.sidebar .sidebar-content {
    background: var(--swiss-light-gray);
}

/* Footer */
.swiss-footer {
    background: var(--swiss-dark);
    color: white;
    padding: 20px;
    margin-top: 40px;
    text-align: center;
    font-size: 0.85rem;
}

/* Loading Animation */
.loading-dots {
    display: inline-flex;
    gap: 4px;
}

.loading-dots span {
    width: 8px;
    height: 8px;
    background: var(--swiss-red);
    border-radius: 50%;
    animation: bounce 1.4s infinite ease-in-out both;
}

.loading-dots span:nth-child(1) { animation-delay: -0.32s; }
.loading-dots span:nth-child(2) { animation-delay: -0.16s; }

@keyframes bounce {
    0%, 80%, 100% { transform: scale(0); }
    40% { transform: scale(1); }
}

/* Responsive Design */
@media (max-width: 768px) {
    .swiss-header h1 { font-size: 1.4rem !important; }
    .result-card { padding: 15px; }
}
</style>
"""
