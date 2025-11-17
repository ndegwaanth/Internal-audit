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
        rating_columns = ['Impact / Consequence Rating', 'Probability / Likelihood Rating']
        
        for col in rating_columns:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().str.title()
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def create_risk_matrix(df):
    # Define the order for ratings
    impact_order = ['Negligible', 'Marginal', 'Serious', 'Critical', 'Catastrophic']
    probability_order = ['Rare', 'Unlikely', 'Moderate', 'Likely', 'Almost Certain']
    
    # Create count matrix
    matrix_data = []
    for impact in impact_order:
        row = []
        for probability in probability_order:
            count = len(df[
                (df['Impact / Consequence Rating'] == impact) & 
                (df['Probability / Likelihood Rating'] == probability)
            ])
            row.append(count)
        matrix_data.append(row)
    
    # Create heatmap
    fig = px.imshow(
        matrix_data,
        x=probability_order,
        y=impact_order,
        labels=dict(x="Probability", y="Impact", color="Number of Risks"),
        color_continuous_scale="RdYlGn_r",  # Red to Green (reversed for higher risk = red)
        aspect="auto"
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
        title="Risk Assessment Matrix Heatmap",
        xaxis_title="Probability / Likelihood Rating",
        yaxis_title="Impact / Consequence Rating",
        height=600
    )
    
    return fig

def create_bubble_chart(df):
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
        return go.Figure()
    
    # Create bubble chart
    fig = px.scatter(
        bubble_df,
        x='Probability_Numeric',
        y='Impact_Numeric',
        size='Count',
        size_max=50,
        hover_data=['Impact', 'Probability', 'Count', 'Risk_Ids'],
        color='Count',
        color_continuous_scale='RdYlGn_r',
        labels={
            'Probability_Numeric': 'Probability',
            'Impact_Numeric': 'Impact',
            'Count': 'Number of Risks'
        }
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
        title="Risk Assessment Bubble Chart",
        xaxis_title="Probability / Likelihood Rating",
        yaxis_title="Impact / Consequence Rating",
        height=600,
        showlegend=False
    )
    
    return fig

def create_bar_charts(df):
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
    
    fig.add_trace(
        go.Bar(
            x=impact_counts.index,
            y=impact_counts.values,
            name='Impact',
            marker_color='indianred'
        ),
        row=1, col=1
    )
    
    # Probability distribution
    probability_counts = df['Probability / Likelihood Rating'].value_counts()
    # Reindex to maintain order
    probability_order = ['Rare', 'Unlikely', 'Moderate', 'Likely', 'Almost Certain']
    probability_counts = probability_counts.reindex([x for x in probability_order if x in probability_counts.index], fill_value=0)
    
    fig.add_trace(
        go.Bar(
            x=probability_counts.index,
            y=probability_counts.values,
            name='Probability',
            marker_color='lightseagreen'
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        title_text="Risk Distribution Analysis",
        height=500,
        showlegend=False
    )
    
    fig.update_xaxes(tickangle=45)
    
    return fig

def create_treemap(df):
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
        return go.Figure()
    
    # Create risk level counts
    risk_level_counts = treemap_df.groupby(['Impact', 'Probability', 'Risk_Level']).size().reset_index(name='Count')
    
    # Create treemap
    fig = px.treemap(
        risk_level_counts,
        path=['Impact', 'Probability'],
        values='Count',
        color='Impact',
        color_continuous_scale='RdYlGn_r',
        title="Risk Distribution Treemap"
    )
    
    fig.update_layout(height=600)
    
    return fig

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
            
            # Display basic info
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Risks", len(df))
            with col2:
                high_impact = len(df[df['Impact / Consequence Rating'].isin(['Critical', 'Catastrophic'])])
                st.metric("High Impact Risks", high_impact)
            with col3:
                high_prob = len(df[df['Probability / Likelihood Rating'].isin(['Likely', 'Almost Certain'])])
                st.metric("High Probability Risks", high_prob)
            with col4:
                critical_risks = len(df[
                    (df['Impact / Consequence Rating'].isin(['Critical', 'Catastrophic'])) & 
                    (df['Probability / Likelihood Rating'].isin(['Likely', 'Almost Certain']))
                ])
                st.metric("Critical Risks", critical_risks)
            
            st.markdown("---")
            
            # Filters
            st.sidebar.header("🔍 Filters")
            
            # Impact filter
            impact_options = ['All'] + sorted(df['Impact / Consequence Rating'].unique())
            selected_impact = st.sidebar.selectbox(
                "Impact Level",
                impact_options
            )
            
            # Probability filter
            probability_options = ['All'] + sorted(df['Probability / Likelihood Rating'].unique())
            selected_probability = st.sidebar.selectbox(
                "Probability Level",
                probability_options
            )
            
            # Apply filters
            filtered_df = df.copy()
            if selected_impact != 'All':
                filtered_df = filtered_df[filtered_df['Impact / Consequence Rating'] == selected_impact]
            if selected_probability != 'All':
                filtered_df = filtered_df[filtered_df['Probability / Likelihood Rating'] == selected_probability]
            
            st.sidebar.info(f"Showing {len(filtered_df)} of {len(df)} risks")
            
            # Visualizations
            tab1, tab2, tab3, tab4, tab5 = st.tabs(["Heat Map", "Bubble Chart", "Bar Charts", "Treemap", "Risk Details"])
            
            with tab1:
                st.plotly_chart(create_risk_matrix(filtered_df), use_container_width=True)
                
            with tab2:
                st.plotly_chart(create_bubble_chart(filtered_df), use_container_width=True)
                
            with tab3:
                st.plotly_chart(create_bar_charts(filtered_df), use_container_width=True)
                
            with tab4:
                st.plotly_chart(create_treemap(filtered_df), use_container_width=True)
                
            with tab5:
                # Display risk details
                if len(filtered_df) > 0:
                    display_columns = ['Risk ID', 'RISKS', 'Impact / Consequence Rating', 
                                     'Probability / Likelihood Rating', 'Inherent Risk Rating', 
                                     'Risk Classification']
                    
                    # Select only columns that exist in the dataframe
                    available_columns = [col for col in display_columns if col in filtered_df.columns]
                    
                    st.dataframe(
                        filtered_df[available_columns],
                        use_container_width=True,
                        height=400
                    )
                    
                    # Download filtered data
                    csv = filtered_df[available_columns].to_csv(index=False)
                    st.download_button(
                        label="📥 Download Filtered Risks as CSV",
                        data=csv,
                        file_name="filtered_risks.csv",
                        mime="text/csv"
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
                for impact, count in impact_summary.items():
                    st.write(f"- {impact}: {count} risks")
            
            with col2:
                st.write("**Probability Distribution:**")
                prob_summary = filtered_df['Probability / Likelihood Rating'].value_counts()
                for prob, count in prob_summary.items():
                    st.write(f"- {prob}: {count} risks")
                    
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
        - **Risk Classification**: Risk category
        """)

if __name__ == "__main__":
    main()