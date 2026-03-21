"""
Credit Risk Analytics Dashboard
Interactive Plotly Dash dashboard — serves as a Power BI-style presentation layer.
Run: python dashboard/app.py
"""
import pickle, os, sys
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Dash, html, dcc, callback, Input, Output
import dash_bootstrap_components as dbc

# ---------------------------------------------------------------------------
# Load artifacts from notebook
# ---------------------------------------------------------------------------
ARTIFACT_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'model_artifacts.pkl')
if not os.path.exists(ARTIFACT_PATH):
    sys.exit("Run the notebook first to generate data/model_artifacts.pkl")

with open(ARTIFACT_PATH, 'rb') as f:
    A = pickle.load(f)

# ---------------------------------------------------------------------------
# Pre-build figures
# ---------------------------------------------------------------------------
COLORS = {'good': '#2ecc71', 'bad': '#e74c3c', 'blue': '#3498db',
          'orange': '#f39c12', 'yellow': '#f1c40f', 'grey': '#95a5a6'}
TIER_COLORS = [COLORS['bad'], COLORS['orange'], COLORS['yellow'], COLORS['good'], COLORS['blue']]

# KPI values
total_borrowers = len(A['scores_test'])
bad_rate = float(A['y_test'].mean() * 100)
auc = A['auc_test']
ks = A['ks_stat']
gini = A['gini']

# --- Fig: ROC ---
fig_roc = go.Figure()
fig_roc.add_trace(go.Scatter(x=A['fpr_train'], y=A['tpr_train'],
                             name=f"Train AUC={A['auc_train']:.3f}", line=dict(color=COLORS['blue'])))
fig_roc.add_trace(go.Scatter(x=A['fpr_test'], y=A['tpr_test'],
                             name=f"Test AUC={A['auc_test']:.3f}", line=dict(color=COLORS['bad'])))
fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name='Random', line=dict(dash='dash', color='grey')))
fig_roc.update_layout(title='ROC Curve', xaxis_title='FPR', yaxis_title='TPR',
                      template='plotly_white', height=400, margin=dict(t=40, b=30))

# --- Fig: Score distribution ---
fig_score = go.Figure()
fig_score.add_trace(go.Histogram(x=A['scores_test'][A['y_test'] == 0], name='Good',
                                 marker_color=COLORS['good'], opacity=0.7, nbinsx=80))
fig_score.add_trace(go.Histogram(x=A['scores_test'][A['y_test'] == 1], name='Bad',
                                 marker_color=COLORS['bad'], opacity=0.7, nbinsx=80))
fig_score.update_layout(barmode='overlay', title='Score Distribution',
                        xaxis_title='Credit Score', template='plotly_white',
                        height=400, margin=dict(t=40, b=30))

# --- Fig: IV ---
fig_iv = px.bar(A['iv_df'], x='IV', y='Feature', orientation='h', color='Predictive_Power',
                color_discrete_map={'Useless': COLORS['grey'], 'Weak': COLORS['orange'],
                                    'Medium': COLORS['blue'], 'Strong': COLORS['good'],
                                    'Very Strong': COLORS['bad']})
fig_iv.update_layout(title='Information Value', template='plotly_white',
                     yaxis={'categoryorder': 'total ascending'}, height=400, margin=dict(t=40, b=30))

# --- Fig: Bad rate by age ---
fig_age = px.bar(A['df_age'], x='age_bucket', y='bad_rate_pct', text='bad_rate_pct',
                 color='bad_rate_pct', color_continuous_scale='RdYlGn_r')
fig_age.update_layout(title='Bad Rate by Age', template='plotly_white',
                      xaxis_title='Age', yaxis_title='Bad Rate %', height=350, margin=dict(t=40, b=30))
fig_age.update_traces(textposition='outside')

# --- Fig: Bad rate by utilization ---
fig_util = px.bar(A['df_util'], x='utilization_band', y='bad_rate_pct', text='bad_rate_pct',
                  color='bad_rate_pct', color_continuous_scale='RdYlGn_r')
fig_util.update_layout(title='Bad Rate by Utilization', template='plotly_white',
                       xaxis_title='Utilization Band', yaxis_title='Bad Rate %',
                       height=350, margin=dict(t=40, b=30))
fig_util.update_traces(textposition='outside')

# --- Fig: Delinquency cascade ---
fig_dlq = px.bar(A['df_dlq'], x='delinquency_status', y='bad_rate_pct', text='bad_rate_pct',
                 color='bad_rate_pct', color_continuous_scale='RdYlGn_r')
fig_dlq.update_layout(title='Bad Rate by Delinquency History', template='plotly_white',
                      xaxis_title='', yaxis_title='Bad Rate %', height=350, margin=dict(t=40, b=30))
fig_dlq.update_traces(textposition='outside')

# --- Fig: Tier summary ---
ts = A['tier_summary']
fig_tier = make_subplots(rows=1, cols=2,
                         subplot_titles=['Portfolio % by Tier', 'Bad Rate % by Tier'])
fig_tier.add_trace(go.Bar(x=ts['tier'], y=ts['pct_portfolio'], marker_color=TIER_COLORS), row=1, col=1)
fig_tier.add_trace(go.Bar(x=ts['tier'], y=ts['bad_rate_pct'], marker_color=TIER_COLORS), row=1, col=2)
fig_tier.update_layout(template='plotly_white', showlegend=False, height=380, margin=dict(t=40, b=30))

# ---------------------------------------------------------------------------
# App layout
# ---------------------------------------------------------------------------
app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY],
           meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}])
app.title = "Credit Risk Dashboard"
server = app.server  # for gunicorn

def kpi_card(title, value, color="#2c3e50"):
    return dbc.Card(
        dbc.CardBody([
            html.P(title, className="text-muted mb-1", style={"fontSize": "0.85rem"}),
            html.H4(value, className="mb-0", style={"color": color, "fontWeight": "700"})
        ]),
        className="shadow-sm", style={"borderTop": f"3px solid {color}"}
    )

app.layout = dbc.Container([
    # Header
    dbc.Row(dbc.Col(html.H3("Credit Risk Analytics Dashboard",
                             className="text-center my-3",
                             style={"fontWeight": "700", "color": "#2c3e50"}))),
    html.Hr(className="mt-0"),

    # KPI row
    dbc.Row([
        dbc.Col(kpi_card("Test Borrowers", f"{total_borrowers:,}", COLORS['blue']), md=2),
        dbc.Col(kpi_card("Bad Rate", f"{bad_rate:.2f}%", COLORS['bad']), md=2),
        dbc.Col(kpi_card("AUC", f"{auc:.4f}", COLORS['good']), md=2),
        dbc.Col(kpi_card("KS Statistic", f"{ks:.4f}", COLORS['orange']), md=2),
        dbc.Col(kpi_card("Gini", f"{gini:.4f}", COLORS['blue']), md=2),
        dbc.Col(kpi_card("Scorecard PDO", "20 pts", COLORS['grey']), md=2),
    ], className="mb-3 g-2"),

    # Tabs
    dbc.Tabs([
        # ---- TAB 1: Model Performance ----
        dbc.Tab(label="Model Performance", children=[
            dbc.Row([
                dbc.Col(dcc.Graph(figure=fig_roc), md=6),
                dbc.Col(dcc.Graph(figure=fig_score), md=6),
            ], className="mt-2"),
            dbc.Row([
                dbc.Col(dcc.Graph(figure=fig_iv), md=12),
            ]),
        ]),

        # ---- TAB 2: Portfolio Analysis ----
        dbc.Tab(label="Portfolio Analysis", children=[
            dbc.Row([
                dbc.Col(dcc.Graph(figure=fig_age), md=4),
                dbc.Col(dcc.Graph(figure=fig_util), md=4),
                dbc.Col(dcc.Graph(figure=fig_dlq), md=4),
            ], className="mt-2"),
        ]),

        # ---- TAB 3: Credit Strategy (interactive) ----
        dbc.Tab(label="Credit Strategy", children=[
            dbc.Row([
                dbc.Col([
                    html.Label("Score Cut-off", className="fw-bold mt-3"),
                    dcc.Slider(id='cutoff-slider', min=400, max=680, step=10, value=530,
                               marks={i: str(i) for i in range(400, 700, 40)},
                               tooltip={"placement": "bottom", "always_visible": True}),
                ], md=12),
            ]),
            dbc.Row([
                dbc.Col(id='strategy-kpis', md=12),
            ], className="mt-2"),
            dbc.Row([
                dbc.Col(dcc.Graph(id='strategy-chart'), md=7),
                dbc.Col(dcc.Graph(figure=fig_tier), md=5),
            ]),
        ]),
    ], className="mt-2"),

    # Footer
    html.Hr(),
    html.P("Case Study — Senior Credit Risk Analyst | Built with Python, Plotly & Dash",
           className="text-center text-muted", style={"fontSize": "0.8rem"}),

], fluid=True)

# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------
@callback(
    [Output('strategy-kpis', 'children'),
     Output('strategy-chart', 'figure')],
    Input('cutoff-slider', 'value')
)
def update_strategy(cutoff):
    approved = A['scores_test'] >= cutoff
    n_app = int(approved.sum())
    app_rate = n_app / total_borrowers * 100
    br = float(A['y_test'][approved].mean() * 100) if n_app > 0 else 0

    kpis = dbc.Row([
        dbc.Col(kpi_card("Cut-off", str(cutoff), COLORS['blue']), md=3),
        dbc.Col(kpi_card("Approved", f"{n_app:,} ({app_rate:.1f}%)", COLORS['good']), md=3),
        dbc.Col(kpi_card("Declined", f"{total_borrowers - n_app:,}", COLORS['bad']), md=3),
        dbc.Col(kpi_card("Bad Rate (approved)", f"{br:.2f}%", COLORS['orange']), md=3),
    ], className="g-2")

    # Cutoff chart with marker
    sdf = A['strat_df']
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=sdf['cutoff'], y=sdf['approval_rate'],
                             name='Approval %', line=dict(color=COLORS['blue'], width=3)), secondary_y=False)
    fig.add_trace(go.Scatter(x=sdf['cutoff'], y=sdf['bad_rate'],
                             name='Bad Rate %', line=dict(color=COLORS['bad'], width=3)), secondary_y=True)
    fig.add_vline(x=cutoff, line_dash="dash", line_color=COLORS['orange'],
                  annotation_text=f"Cut-off = {cutoff}")
    fig.update_layout(title='Approval vs Bad Rate by Cut-off', xaxis_title='Score',
                      template='plotly_white', height=380, margin=dict(t=40, b=30))
    fig.update_yaxes(title_text='Approval %', secondary_y=False)
    fig.update_yaxes(title_text='Bad Rate %', secondary_y=True)

    return kpis, fig


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8050))
    app.run(debug=False, host='0.0.0.0', port=port)
