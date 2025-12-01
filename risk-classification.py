import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# Load and process the data
@st.cache_data
def load_data(uploaded_file):
    try:
        # Try reading the Excel file
        df = pd.read_excel(uploaded_file, sheet_name='Sheet1')
        
        # Clean column names
        df.columns = [col.strip() for col in df.columns]
        
        # Standardize rating columns
        rating_columns = ['Impact / Consequence Rating', 'Probability / Likelihood Rating', 'Risk Classification']
        
        for col in rating_columns:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

# Define color themes
THEMES = {
    "Red-Yellow-Green (Risk Matrix)": "RdYlGn_r",
    "Viridis (High Contrast)": "viridis",
    "Plasma (Warm)": "plasma",
    "Inferno (Hot)": "inferno",
    "Blues (Cool)": "blues"
}

def create_risk_matrix(df, theme):
    # Define the order for ratings
    impact_order = ['Negligible', 'Marginal', 'Serious', 'Critical', 'Catastrophic']
    probability_order = ['Rare', 'Unlikely', 'Moderate', 'Likely', 'Almost Certain']
    
    # Create count matrix and risk ID matrix
    matrix_data = []
    risk_ids_matrix = []
    
    for impact in impact_order:
        row_counts = []
        row_risk_ids = []
        for probability in probability_order:
            risks = df[
                (df['Impact / Consequence Rating'] == impact) & 
                (df['Probability / Likelihood Rating'] == probability)
            ]
            count = len(risks)
            risk_ids = ', '.join(risks['Risk ID'].astype(str)) if len(risks) > 0 else 'None'
            row_counts.append(count)
            row_risk_ids.append(risk_ids)
        matrix_data.append(row_counts)
        risk_ids_matrix.append(row_risk_ids)
    
    # Create custom hover text
    hover_text = []
    for i, impact in enumerate(impact_order):
        hover_row = []
        for j, probability in enumerate(probability_order):
            count = matrix_data[i][j]
            risk_ids = risk_ids_matrix[i][j]
            if count > 0:
                hover_text.append(f"<b>Impact:</b> {impact}<br><b>Probability:</b> {probability}<br><b>Number of Risks:</b> {count}<br><b>Risk IDs:</b> {risk_ids}")
            else:
                hover_text.append(f"<b>Impact:</b> {impact}<br><b>Probability:</b> {probability}<br><b>Number of Risks:</b> 0<br><b>Risk IDs:</b> None")
        #hover_text.append(hover_row)
    
    # Reshape hover text to match matrix shape
    hover_text_2d = []
    for i in range(len(impact_order)):
        start_idx = i * len(probability_order)
        end_idx = start_idx + len(probability_order)
        hover_text_2d.append(hover_text[start_idx:end_idx])
    
    # Create heatmap with custom hover data
    fig = px.imshow(
        matrix_data,
        x=probability_order,
        y=impact_order,
        labels=dict(x="Probability", y="Impact", color="Number of Risks"),
        color_continuous_scale=THEMES[theme],
        aspect="auto"
    )
    
    # Update hover template
    fig.update_traces(
        hovertemplate='%{customdata}<extra></extra>',
        customdata=hover_text_2d
    )
    
    # Add annotations
    for i in range(len(impact_order)):
        for j in range(len(probability_order)):
            fig.add_annotation(
                x=j, y=i,
                text=str(matrix_data[i][j]),
                showarrow=False,
                font=dict(color="white" if matrix_data[i][j] > (max(map(max, matrix_data)) / 2) else "black", size=14)
            )
    
    fig.update_layout(
        title="Risk Assessment Matrix Heatmap<br><sub>Hover over cells to see Risk IDs</sub>",
        xaxis_title="Probability / Likelihood Rating",
        yaxis_title="Impact / Consequence Rating",
        height=600
    )
    
    return fig

def create_bubble_chart(df, theme):
    # Define the order and numeric mapping for ratings
    impact_order = ['Negligible', 'Marginal', 'Serious', 'Critical', 'Catastrophic']
    probability_order = ['Rare', 'Unlikely', 'Moderate', 'Likely', 'Almost Certain']
    
    impact_numeric = {rating: i for i, rating in enumerate(impact_order)}
    probability_numeric = {rating: i for i, rating in enumerate(probability_order)}
    
    # Prepare data for bubble chart
    bubble_data = []
    for impact in impact_order:
        for probability in probability_order:
            risks = df[
                (df['Impact / Consequence Rating'] == impact) & 
                (df['Probability / Likelihood Rating'] == probability)
            ]
            count = len(risks)
            if count > 0:
                bubble_data.append({
                    'Impact': impact,
                    'Probability': probability,
                    'Count': count,
                    'Impact_Numeric': impact_numeric[impact],
                    'Probability_Numeric': probability_numeric[probability],
                    'Risk_Ids': ', '.join(risks['Risk ID'].astype(str))
                })
    
    bubble_df = pd.DataFrame(bubble_data)
    
    if bubble_df.empty:
        # Return empty figure with message
        fig = go.Figure()
        fig.add_annotation(
            text="No data available for selected filters",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False,
            font=dict(size=16)
        )
        fig.update_layout(height=600)
        return fig
    
    # Create bubble chart with enhanced hover
    fig = px.scatter(
        bubble_df,
        x='Probability_Numeric',
        y='Impact_Numeric',
        size='Count',
        size_max=50,
        hover_data={
            'Impact': True,
            'Probability': True,
            'Count': True,
            'Risk_Ids': True,
            'Impact_Numeric': False,
            'Probability_Numeric': False
        },
        color='Count',
        color_continuous_scale=THEMES[theme],
        labels={
            'Probability_Numeric': 'Probability',
            'Impact_Numeric': 'Impact',
            'Count': 'Number of Risks'
        }
    )
    
    # Customize hover template
    fig.update_traces(
        hovertemplate='<b>Impact:</b> %{customdata[0]}<br>' +
                     '<b>Probability:</b> %{customdata[1]}<br>' +
                     '<b>Number of Risks:</b> %{customdata[2]}<br>' +
                     '<b>Risk IDs:</b> %{customdata[3]}<extra></extra>'
    )
    
    # Update axes
    fig.update_xaxes(
        tickvals=list(range(len(probability_order))),
        ticktext=probability_order
    )
    fig.update_yaxes(
        tickvals=list(range(len(impact_order))),
        ticktext=impact_order
    )
    
    fig.update_layout(
        title="Risk Assessment Bubble Chart<br><sub>Hover over bubbles to see Risk IDs</sub>",
        xaxis_title="Probability / Likelihood Rating",
        yaxis_title="Impact / Consequence Rating",
        height=600,
        showlegend=False
    )
    
    return fig

def create_bar_charts(df, theme):
    # Create subplots
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Risks by Impact Level', 'Risks by Probability Level')
    )
    
    # Impact distribution
    impact_counts = df['Impact / Consequence Rating'].value_counts()
    # Reindex to maintain order
    impact_order = ['Negligible', 'Marginal', 'Serious', 'Critical', 'Catastrophic']
    impact_counts = impact_counts.reindex([x for x in impact_order if x in impact_counts.index], fill_value=0)
    
    # Get risk IDs for each impact level
    impact_risk_ids = {}
    for impact in impact_counts.index:
        risks = df[df['Impact / Consequence Rating'] == impact]
        impact_risk_ids[impact] = ', '.join(risks['Risk ID'].astype(str)) if len(risks) > 0 else 'None'
    
    # Safe color calculation
    if len(impact_counts) > 0:
        if len(impact_counts) == 1:
            colors = [px.colors.sample_colorscale(THEMES[theme], [0.5])[0]] * len(impact_counts)
        else:
            colors = px.colors.sample_colorscale(THEMES[theme], [n/(len(impact_counts)-1) for n in range(len(impact_counts))])
    else:
        colors = ['lightgray']
    
    fig.add_trace(
        go.Bar(
            x=impact_counts.index,
            y=impact_counts.values,
            name='Impact',
            marker_color=colors,
            customdata=[impact_risk_ids[impact] for impact in impact_counts.index],
            hovertemplate='<b>Impact:</b> %{x}<br>' +
                         '<b>Number of Risks:</b> %{y}<br>' +
                         '<b>Risk IDs:</b> %{customdata}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Probability distribution
    probability_counts = df['Probability / Likelihood Rating'].value_counts()
    # Reindex to maintain order
    probability_order = ['Rare', 'Unlikely', 'Moderate', 'Likely', 'Almost Certain']
    probability_counts = probability_counts.reindex([x for x in probability_order if x in probability_counts.index], fill_value=0)
    
    # Get risk IDs for each probability level
    probability_risk_ids = {}
    for probability in probability_counts.index:
        risks = df[df['Probability / Likelihood Rating'] == probability]
        probability_risk_ids[probability] = ', '.join(risks['Risk ID'].astype(str)) if len(risks) > 0 else 'None'
    
    # Safe color calculation
    if len(probability_counts) > 0:
        if len(probability_counts) == 1:
            colors_prob = [px.colors.sample_colorscale(THEMES[theme], [0.5])[0]] * len(probability_counts)
        else:
            colors_prob = px.colors.sample_colorscale(THEMES[theme], [n/(len(probability_counts)-1) for n in range(len(probability_counts))])
    else:
        colors_prob = ['lightgray']
    
    fig.add_trace(
        go.Bar(
            x=probability_counts.index,
            y=probability_counts.values,
            name='Probability',
            marker_color=colors_prob,
            customdata=[probability_risk_ids[probability] for probability in probability_counts.index],
            hovertemplate='<b>Probability:</b> %{x}<br>' +
                         '<b>Number of Risks:</b> %{y}<br>' +
                         '<b>Risk IDs:</b> %{customdata}<extra></extra>'
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        title_text="Risk Distribution Analysis<br><sub>Hover over bars to see Risk IDs</sub>",
        height=500,
        showlegend=False
    )
    
    fig.update_xaxes(tickangle=45)
    
    return fig

def create_treemap(df, theme):
    # Create treemap data
    treemap_data = []
    for _, row in df.iterrows():
        treemap_data.append({
            'Risk_ID': row['Risk ID'],
            'Impact': row['Impact / Consequence Rating'],
            'Probability': row['Probability / Likelihood Rating'],
            'Risk_Level': f"{row['Impact / Consequence Rating']} - {row['Probability / Likelihood Rating']}",
            'Count': 1
        })
    
    treemap_df = pd.DataFrame(treemap_data)
    
    if treemap_df.empty:
        # Return empty figure with message
        fig = go.Figure()
        fig.add_annotation(
            text="No data available for selected filters",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False,
            font=dict(size=16)
        )
        fig.update_layout(height=600, title="Risk Distribution Treemap")
        return fig
    
    # Create risk level counts
    risk_level_counts = treemap_df.groupby(['Impact', 'Probability', 'Risk_Level']).size().reset_index(name='Count')
    
    # Get all risk IDs for each combination
    risk_ids_by_combo = {}
    for _, row in risk_level_counts.iterrows():
        impact = row['Impact']
        probability = row['Probability']
        risks = df[
            (df['Impact / Consequence Rating'] == impact) & 
            (df['Probability / Likelihood Rating'] == probability)
        ]
        risk_ids_by_combo[(impact, probability)] = ', '.join(risks['Risk ID'].astype(str))
    
    # Add risk IDs to the dataframe
    risk_level_counts['Risk_Ids'] = risk_level_counts.apply(
        lambda x: risk_ids_by_combo.get((x['Impact'], x['Probability']), 'None'), 
        axis=1
    )
    
    # Create treemap
    fig = px.treemap(
        risk_level_counts,
        path=['Impact', 'Probability'],
        values='Count',
        color='Impact',
        color_continuous_scale=THEMES[theme],
        title="Risk Distribution Treemap<br><sub>Hover over sections to see Risk IDs</sub>",
        custom_data=['Risk_Ids']
    )
    
    # Update hover template
    fig.update_traces(
        hovertemplate='<b>%{label}</b><br>' +
                     'Number of Risks: %{value}<br>' +
                     'Risk IDs: %{customdata[0]}<extra></extra>'
    )
    
    fig.update_layout(height=600)
    
    return fig

def create_risk_details_table(df):
    """Create an expandable risk details table"""
    if len(df) == 0:
        return None
    
    # Select columns to display
    display_columns = ['Risk ID', 'RISKS', 'Impact / Consequence Rating', 
                     'Probability / Likelihood Rating', 'Inherent Risk Rating', 
                     'Risk Classification']
    
    # Select only columns that exist in the dataframe
    available_columns = [col for col in display_columns if col in df.columns]
    
    return df[available_columns]

def main():
    st.set_page_config(page_title="Risk Assessment Dashboard", layout="wide")
    
    st.title("🚨 Business Risk Assessment Dashboard")
    st.markdown("---")
    
    # File upload
    uploaded_file = st.file_uploader("Upload your Risk Register Excel file", type=['xlsx'])
    
    if uploaded_file is not None:
        df = load_data(uploaded_file)
        
        if df is not None:
            st.success(f"✅ Successfully loaded {len(df)} risk records")
            
            # Display basic info with Risk Classification
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Risks", len(df))
            with col2:
                # Count risks by Risk Classification
                if 'Risk Classification' in df.columns:
                    # Get the distribution of risk classifications
                    risk_class_counts = df['Risk Classification'].value_counts()
                    
                    # Create a clean display of the classifications
                    if len(risk_class_counts) > 0:
                        # Show the most common classification as the metric
                        most_common = risk_class_counts.idxmax()
                        count_most_common = risk_class_counts.max()
                        
                        # Create a formatted string for the metric label
                        st.metric(
                            "Risk Classification", 
                            f"{most_common}",
                            help=f"Most common risk classification. Total breakdown: {', '.join([f'{k}: {v}' for k, v in risk_class_counts.items()])}"
                        )
                    else:
                        st.metric("Risk Classification", "No Data")
                else:
                    st.metric("Risk Classification", "Column Not Found")
            with col3:
                high_prob = len(df[df['Probability / Likelihood Rating'].isin(['Likely', 'Almost Certain'])])
                st.metric("High Probability Risks", high_prob)
            with col4:
                critical_risks = len(df[
                    (df['Impact / Consequence Rating'].isin(['Critical', 'Catastrophic'])) & 
                    (df['Probability / Likelihood Rating'].isin(['Likely', 'Almost Certain']))
                ])
                st.metric("Critical Risks", critical_risks)
            
            # Optional: Show detailed breakdown of Risk Classifications
            if 'Risk Classification' in df.columns:
                risk_class_counts = df['Risk Classification'].value_counts()
                if len(risk_class_counts) > 0:
                    with st.expander("📊 Detailed Risk Classification Breakdown", expanded=False):
                        cols = st.columns(min(4, len(risk_class_counts)))
                        for idx, (classification, count) in enumerate(risk_class_counts.items()):
                            with cols[idx % len(cols)]:
                                st.metric(f"{classification} Risks", count)
            
            st.markdown("---")
            
            # Filters and Theme Selection in sidebar
            st.sidebar.header("🎛️ Dashboard Controls")
            
            # Theme selection
            selected_theme = st.sidebar.selectbox(
                "🎨 Select Color Theme",
                list(THEMES.keys()),
                index=0
            )
            
            st.sidebar.markdown("---")
            st.sidebar.header("🔍 Risk Filters")
            
            # Multi-select filters for Impact
            impact_options = sorted(df['Impact / Consequence Rating'].unique())
            selected_impacts = st.sidebar.multiselect(
                "Impact Levels",
                impact_options,
                default=impact_options,
                help="Select one or more impact levels to filter"
            )
            
            # Multi-select filters for Probability
            probability_options = sorted(df['Probability / Likelihood Rating'].unique())
            selected_probabilities = st.sidebar.multiselect(
                "Probability Levels",
                probability_options,
                default=probability_options,
                help="Select one or more probability levels to filter"
            )
            
            # Optional: Filter by Risk Classification
            if 'Risk Classification' in df.columns:
                classification_options = sorted(df['Risk Classification'].unique())
                selected_classifications = st.sidebar.multiselect(
                    "Risk Classifications",
                    classification_options,
                    default=classification_options,
                    help="Select one or more risk classifications to filter"
                )
            
            # Apply filters
            filtered_df = df.copy()
            if selected_impacts:
                filtered_df = filtered_df[filtered_df['Impact / Consequence Rating'].isin(selected_impacts)]
            if selected_probabilities:
                filtered_df = filtered_df[filtered_df['Probability / Likelihood Rating'].isin(selected_probabilities)]
            if 'Risk Classification' in df.columns and 'selected_classifications' in locals():
                filtered_df = filtered_df[filtered_df['Risk Classification'].isin(selected_classifications)]
            
            st.sidebar.info(f"Showing {len(filtered_df)} of {len(df)} risks")
            
            # Reset filters button
            if st.sidebar.button("🔄 Reset All Filters"):
                st.rerun()
            
            # Visualizations
            tab1, tab2, tab3, tab4 = st.tabs(["📊 Heat Map", "🟠 Bubble Chart", "📈 Bar Charts", "🌳 Treemap"])
            
            with tab1:
                st.plotly_chart(create_risk_matrix(filtered_df, selected_theme), use_container_width=True)
                
            with tab2:
                st.plotly_chart(create_bubble_chart(filtered_df, selected_theme), use_container_width=True)
                
            with tab3:
                st.plotly_chart(create_bar_charts(filtered_df, selected_theme), use_container_width=True)
                
            with tab4:
                st.plotly_chart(create_treemap(filtered_df, selected_theme), use_container_width=True)
            
            # Risk Details Section
            st.markdown("---")
            st.subheader("📋 Risk Details")
            
            if len(filtered_df) > 0:
                # Display risk details in an expandable section
                with st.expander(f"View All {len(filtered_df)} Filtered Risks", expanded=True):
                    risk_details = create_risk_details_table(filtered_df)
                    if risk_details is not None:
                        st.dataframe(
                            risk_details,
                            use_container_width=True,
                            height=400
                        )
                        
                        # Download filtered data
                        csv = risk_details.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Filtered Risks as CSV",
                            data=csv,
                            file_name="filtered_risks.csv",
                            mime="text/csv",
                            key="download_csv"
                        )
            else:
                st.info("No risks match the selected filters")
            
            # Risk summary
            st.markdown("---")
            st.subheader("📊 Risk Summary")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Impact Distribution:**")
                impact_summary = filtered_df['Impact / Consequence Rating'].value_counts()
                if len(impact_summary) > 0:
                    for impact, count in impact_summary.items():
                        st.write(f"- {impact}: {count} risks")
                else:
                    st.write("No impact data available")
            
            with col2:
                st.write("**Probability Distribution:**")
                prob_summary = filtered_df['Probability / Likelihood Rating'].value_counts()
                if len(prob_summary) > 0:
                    for prob, count in prob_summary.items():
                        st.write(f"- {prob}: {count} risks")
                else:
                    st.write("No probability data available")
                    
            # Risk Classification Summary
            if 'Risk Classification' in filtered_df.columns:
                st.markdown("---")
                st.subheader("🏷️ Risk Classification Summary")
                classification_summary = filtered_df['Risk Classification'].value_counts()
                if len(classification_summary) > 0:
                    cols = st.columns(min(3, len(classification_summary)))
                    for idx, (classification, count) in enumerate(classification_summary.items()):
                        with cols[idx % len(cols)]:
                            st.metric(f"{classification}", count)
                    
        else:
            st.error("Failed to load data. Please check the file format.")
    else:
        st.info("👆 Please upload an Excel file to begin analysis")
        
        # Display sample of expected format
        st.subheader("Expected Data Format")
        st.write("""
        Your Excel file should contain these key columns:
        - **Risk ID**: Unique identifier for each risk
        - **RISKS**: Description of the risk
        - **Impact / Consequence Rating**: Negligible, Marginal, Serious, Critical, or Catastrophic
        - **Probability / Likelihood Rating**: Rare, Unlikely, Moderate, Likely, or Almost Certain
        - **Inherent Risk Rating**: Calculated risk score
        - **Risk Classification**: Risk category (Acceptable, Action Advisable, Action Required, Unacceptable)
        """)

if __name__ == "__main__":
    main()