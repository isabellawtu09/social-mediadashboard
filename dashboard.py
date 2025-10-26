import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import spearmanr, pearsonr
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings("ignore")

# Page configuration
st.set_page_config(
    page_title="Social Media & Emotions Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
    h1 {
        color: #1f77b4;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

# Load and cache data
@st.cache_data
def load_data():
    """Load and preprocess the training data"""
    train_df = pd.read_csv("socialmedia_traindata.csv", on_bad_lines='skip')
    train_df = train_df.dropna()
    train_df['Age'] = pd.to_numeric(train_df['Age'], errors='coerce')
    train_df['Age'].fillna(train_df['Age'].median(), inplace=True)
    return train_df

@st.cache_data
def perform_clustering(df):
    """Perform K-means clustering on the data"""
    # Filter valid genders
    df_filtered = df[df['Gender'].isin(['Female', 'Male', 'Non-binary'])].copy()
    
    # Select features
    features = df_filtered[['Posts_Per_Day', 'Likes_Received_Per_Day', 
                            'Comments_Received_Per_Day', 'Messages_Sent_Per_Day']]
    
    # Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)
    
    # Perform clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    df_filtered['Cluster'] = kmeans.fit_predict(X_scaled)
    
    return df_filtered

# Load data
try:
    train_df = load_data()
    clustered_df = perform_clustering(train_df)
    data_loaded = True
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.info("Please make sure the CSV file is in the correct location")
    data_loaded = False

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/bar-chart.png", width=100)
    st.title("📊 Navigation")
    st.markdown("---")
    
    if data_loaded:
        st.metric("Total Records", f"{len(train_df):,}")
        st.metric("Age Range", f"{int(train_df['Age'].min())} - {int(train_df['Age'].max())}")
        st.metric("Unique Emotions", train_df['Dominant_Emotion'].nunique())
        st.metric("Platforms", train_df['Platform'].nunique())
    
    st.markdown("---")
    st.markdown("""
    ### About
    This dashboard analyzes social media usage patterns 
    and their relationship with dominant emotions.
    
    **Key Insights:**
    - Usage patterns by emotion
    - Platform preferences
    - User clustering
    - Engagement metrics
    """)

# Main content
if data_loaded:
    st.title("📱 Social Media Usage & Emotions Analysis Dashboard")
    st.markdown("### Comprehensive EDA on Social Media Behavior and Emotional States")
    st.markdown("---")
    
    # Create tabs for different sections
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📈 Overview", 
        "🎭 Emotions Analysis", 
        "📱 Platform Insights", 
        "👥 Demographics",
        "🔍 Clustering Analysis",
        "📊 Statistical Insights"
    ])
    
    # TAB 1: OVERVIEW
    with tab1:
        st.header("Dataset Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Users", f"{len(train_df):,}", delta=None)
        with col2:
            avg_usage = train_df['Daily_Usage_Time (minutes)'].mean()
            st.metric("Avg Daily Usage", f"{avg_usage:.1f} min")
        with col3:
            avg_posts = train_df['Posts_Per_Day'].mean()
            st.metric("Avg Posts/Day", f"{avg_posts:.1f}")
        with col4:
            avg_likes = train_df['Likes_Received_Per_Day'].mean()
            st.metric("Avg Likes/Day", f"{avg_likes:.1f}")
        
        st.markdown("---")
        
        # Data preview
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Sample Data")
            st.dataframe(train_df.head(10), use_container_width=True)
        
        with col2:
            st.subheader("Data Summary")
            st.write(f"**Rows:** {train_df.shape[0]:,}")
            st.write(f"**Columns:** {train_df.shape[1]}")
            st.write(f"**Missing Values:** {train_df.isnull().sum().sum()}")
            
            st.subheader("Emotions Distribution")
            emotion_counts = train_df['Dominant_Emotion'].value_counts()
            fig = px.pie(values=emotion_counts.values, names=emotion_counts.index, 
                        hole=0.4, color_discrete_sequence=px.colors.qualitative.Pastel)
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Distribution plots
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Daily Usage Time Distribution")
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.histplot(train_df['Daily_Usage_Time (minutes)'], kde=True, bins=30, color='skyblue', ax=ax)
            ax.set_xlabel('Usage Time (minutes)')
            ax.set_ylabel('Frequency')
            ax.set_title('Distribution of Social Media Usage')
            st.pyplot(fig)
        
        with col2:
            st.subheader("Age Distribution")
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.histplot(train_df['Age'], kde=True, bins=20, color='coral', ax=ax)
            ax.set_xlabel('Age')
            ax.set_ylabel('Frequency')
            ax.set_title('Age Distribution of Users')
            st.pyplot(fig)
    
    # TAB 2: EMOTIONS ANALYSIS
    with tab2:
        st.header("🎭 Emotion-Based Analysis")
        
        # Calculate metrics by emotion
        emotions = ['Happiness', 'Anxiety', 'Anger', 'Sadness', 'Neutral', 'Boredom']
        emotion_data = {}
        
        for emotion in emotions:
            users = train_df[train_df['Dominant_Emotion'] == emotion]
            emotion_data[emotion] = {
                'usage': users['Daily_Usage_Time (minutes)'].mean(),
                'posts': users['Posts_Per_Day'].mean(),
                'likes': users['Likes_Received_Per_Day'].mean(),
                'comments': users['Comments_Received_Per_Day'].mean(),
                'messages': users['Messages_Sent_Per_Day'].mean()
            }
        
        # Average Daily Usage by Emotion
        st.subheader("Average Daily Usage Time by Emotion")
        col1, col2 = st.columns([2, 1])
        
        with col1:
            usage_data = [emotion_data[e]['usage'] for e in emotions]
            colors_map = {
                'Happiness': '#ec96ce',
                'Anxiety': '#b2f134',
                'Anger': '#e15e2c',
                'Sadness': '#1E90FF',
                'Neutral': '#f1f592',
                'Boredom': '#c063e6'
            }
            colors = [colors_map[e] for e in emotions]
            
            fig = go.Figure(data=[
                go.Bar(x=emotions, y=usage_data, marker_color=colors,
                      text=[f'{v:.1f}' for v in usage_data],
                      textposition='outside')
            ])
            fig.update_layout(
                title="Average Daily Usage Time by Emotion",
                xaxis_title="Emotion",
                yaxis_title="Usage Time (minutes)",
                height=400,
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### Key Findings")
            max_emotion = emotions[usage_data.index(max(usage_data))]
            min_emotion = emotions[usage_data.index(min(usage_data))]
            st.info(f"""
            **Highest Usage:**  
            {max_emotion}: {max(usage_data):.1f} min
            
            **Lowest Usage:**  
            {min_emotion}: {min(usage_data):.1f} min
            
            **Correlation:**  
            0.996 between usage time and likes received
            """)
        
        st.markdown("---")
        
        # Engagement metrics by emotion
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Average Posts Per Day by Emotion")
            posts_data = [emotion_data[e]['posts'] for e in emotions]
            fig = go.Figure(data=[
                go.Bar(x=emotions, y=posts_data, marker_color=colors)
            ])
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Average Likes Received by Emotion")
            likes_data = [emotion_data[e]['likes'] for e in emotions]
            fig = go.Figure(data=[
                go.Bar(x=emotions, y=likes_data, marker_color=colors)
            ])
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Comparison: Happy vs Sad users
        st.subheader("Comparative Analysis: Happy vs Sad Users")
        
        happy_users = train_df[train_df['Dominant_Emotion'] == 'Happiness']
        sad_users = train_df[train_df['Dominant_Emotion'] == 'Sadness']
        
        comparison_metrics = {
            'Metric': ['Average Usage (min)', 'Total Usage (min)', 'Min Usage', 'Max Usage', 'Avg Posts', 'Avg Likes'],
            'Happy Users': [
                happy_users['Daily_Usage_Time (minutes)'].mean(),
                happy_users['Daily_Usage_Time (minutes)'].sum(),
                happy_users['Daily_Usage_Time (minutes)'].min(),
                happy_users['Daily_Usage_Time (minutes)'].max(),
                happy_users['Posts_Per_Day'].mean(),
                happy_users['Likes_Received_Per_Day'].mean()
            ],
            'Sad Users': [
                sad_users['Daily_Usage_Time (minutes)'].mean(),
                sad_users['Daily_Usage_Time (minutes)'].sum(),
                sad_users['Daily_Usage_Time (minutes)'].min(),
                sad_users['Daily_Usage_Time (minutes)'].max(),
                sad_users['Posts_Per_Day'].mean(),
                sad_users['Likes_Received_Per_Day'].mean()
            ]
        }
        
        comparison_df = pd.DataFrame(comparison_metrics)
        st.dataframe(comparison_df.style.format({'Happy Users': '{:.2f}', 'Sad Users': '{:.2f}'}), 
                    use_container_width=True)
        
        st.success("""
        **Key Insight:** Users who reported feeling happy posted more frequently and received more likes 
        compared to users feeling sad. This suggests that positive emotional states correlate with 
        increased engagement and social validation on social media platforms.
        """)
    
    # TAB 3: PLATFORM INSIGHTS
    with tab3:
        st.header("📱 Platform Preference Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Platform Distribution")
            platform_counts = train_df['Platform'].value_counts()
            fig = px.pie(values=platform_counts.values, 
                        names=platform_counts.index,
                        hole=0.4,
                        color_discrete_sequence=px.colors.qualitative.Set3)
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Platform Usage by Age")
            age_platform = train_df.groupby(['Platform', 'Age']).size().reset_index(name='Count')
            fig = px.bar(age_platform, x='Age', y='Count', color='Platform',
                        barmode='stack', color_discrete_sequence=px.colors.qualitative.Pastel)
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Emotion distribution by platform
        st.subheader("Emotion Distribution by Platform")
        emotion_by_platform = pd.crosstab(train_df['Platform'], train_df['Dominant_Emotion'])
        
        fig, ax = plt.subplots(figsize=(14, 8))
        sns.heatmap(emotion_by_platform, annot=True, cmap="YlGnBu", fmt="d", 
                   cbar_kws={'label': 'Number of Users'}, ax=ax)
        ax.set_title('Emotion Distribution by Platform', fontsize=16, pad=20)
        ax.set_xlabel('Emotion', fontsize=12)
        ax.set_ylabel('Platform', fontsize=12)
        st.pyplot(fig)
        
        st.markdown("---")
        
        # Platform engagement metrics
        st.subheader("Average Engagement by Platform")
        
        platform_engagement = train_df.groupby('Platform').agg({
            'Daily_Usage_Time (minutes)': 'mean',
            'Posts_Per_Day': 'mean',
            'Likes_Received_Per_Day': 'mean',
            'Comments_Received_Per_Day': 'mean'
        }).round(2)
        
        st.dataframe(platform_engagement, use_container_width=True)
    
    # TAB 4: DEMOGRAPHICS
    with tab4:
        st.header("👥 Demographic Analysis")
        
        # Filter valid genders
        df_clean = train_df[train_df['Gender'].isin(['Female', 'Male', 'Non-binary'])]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Gender Distribution")
            gender_counts = df_clean['Gender'].value_counts()
            fig = px.bar(x=gender_counts.index, y=gender_counts.values,
                        color=gender_counts.index,
                        color_discrete_map={
                            'Female': '#ff33bb',
                            'Male': '#009999',
                            'Non-binary': '#666699'
                        })
            fig.update_layout(showlegend=False, height=350)
            fig.update_xaxes(title="Gender")
            fig.update_yaxes(title="Count")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Gender vs Average Usage Time")
            fig, ax = plt.subplots(figsize=(8, 6))
            custom_palette = {
                'Female': '#ff33bb',
                'Male': '#009999',
                'Non-binary': '#666699'
            }
            sns.violinplot(x='Gender', y='Daily_Usage_Time (minutes)', 
                          data=df_clean, palette=custom_palette, ax=ax)
            ax.set_title('Gender vs. Average Usage Time', fontsize=14)
            st.pyplot(fig)
        
        st.markdown("---")
        
        # Gender and Emotion heatmap
        st.subheader("Emotion Distribution by Gender")
        
        gender_emotion_heatmap = pd.crosstab(df_clean['Gender'], df_clean['Dominant_Emotion'])
        
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.heatmap(gender_emotion_heatmap, annot=True, cmap="BuPu", fmt="d", ax=ax)
        ax.set_title('Emotion Distribution by Gender', fontsize=16)
        ax.set_xlabel('Emotion', fontsize=12)
        ax.set_ylabel('Gender', fontsize=12)
        st.pyplot(fig)
        
        st.markdown("---")
        
        # Age vs Usage analysis
        st.subheader("Age vs Daily Usage Time")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.regplot(x='Age', y='Daily_Usage_Time (minutes)', data=train_df,
                       scatter_kws={'s':10, 'alpha':0.5}, 
                       line_kws={'color':'red'}, ax=ax)
            ax.set_title('Age vs Social Media Usage (with regression line)')
            ax.set_xlabel('Age')
            ax.set_ylabel('Daily Usage Time (minutes)')
            st.pyplot(fig)
        
        with col2:
            st.markdown("### Correlation Analysis")
            df_clean_age = train_df.dropna(subset=['Age', 'Daily_Usage_Time (minutes)'])
            correlation, p_value = spearmanr(df_clean_age['Age'], 
                                             df_clean_age['Daily_Usage_Time (minutes)'])
            
            st.metric("Spearman Correlation", f"{correlation:.3f}")
            st.metric("P-value", f"{p_value:.4f}")
            
            if p_value > 0.05:
                st.warning("⚠️ No significant correlation between age and usage time (p > 0.05)")
            else:
                st.success("✅ Significant correlation found")
    
    # TAB 5: CLUSTERING ANALYSIS
    with tab5:
        st.header("🔍 User Clustering Analysis")
        
        st.info("""
        K-means clustering was performed using engagement features: Posts, Likes, Comments, and Messages per day.
        Users were grouped into 3 distinct clusters based on their social media behavior patterns.
        """)
        
        # Define consistent cluster colors and characteristics
        cluster_colors = {
            0: '#2ecc71',  # Green - Happy Cluster
            1: '#3498db',  # Blue - Neutral/Bored Cluster
            2: '#e74c3c'   # Red - Angry/Mixed Emotions Cluster
        }
        
        cluster_names = {
            0: 'Cluster 0: Happy Users (72% Happiness)',
            1: 'Cluster 1: Neutral/Bored Users (29% Neutral, 27% Bored)',
            2: 'Cluster 2: Angry/Mixed Users (30% Anger)'
        }
        
        # Display cluster color legend
        st.markdown("### 🎨 Cluster Color Guide")
        col_legend1, col_legend2, col_legend3 = st.columns(3)
        
        with col_legend1:
            st.markdown(f"""
            <div style='background-color: {cluster_colors[0]}; padding: 15px; border-radius: 10px; text-align: center;'>
                <h4 style='color: white; margin: 0;'>🟢 Cluster 0</h4>
                <p style='color: white; margin: 5px 0 0 0; font-size: 14px;'>Happy Users</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col_legend2:
            st.markdown(f"""
            <div style='background-color: {cluster_colors[1]}; padding: 15px; border-radius: 10px; text-align: center;'>
                <h4 style='color: white; margin: 0;'>🔵 Cluster 1</h4>
                <p style='color: white; margin: 5px 0 0 0; font-size: 14px;'>Neutral/Bored Users</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col_legend3:
            st.markdown(f"""
            <div style='background-color: {cluster_colors[2]}; padding: 15px; border-radius: 10px; text-align: center;'>
                <h4 style='color: white; margin: 0;'>🔴 Cluster 2</h4>
                <p style='color: white; margin: 5px 0 0 0; font-size: 14px;'>Angry/Mixed Users</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Cluster Distribution")
            cluster_counts = clustered_df['Cluster'].value_counts().sort_index()
            
            # Create bar chart with custom colors
            fig = go.Figure(data=[
                go.Bar(x=[f'Cluster {i}' for i in cluster_counts.index],
                       y=cluster_counts.values,
                       marker_color=[cluster_colors[i] for i in cluster_counts.index],
                       text=cluster_counts.values,
                       textposition='outside')
            ])
            fig.update_layout(
                xaxis_title="Cluster",
                yaxis_title="Number of Users",
                showlegend=False,
                height=350
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Cluster Characteristics")
            cluster_means = clustered_df.groupby('Cluster')[['Age', 'Daily_Usage_Time (minutes)',
                                                              'Posts_Per_Day', 'Likes_Received_Per_Day']].mean()
            st.dataframe(cluster_means.round(2), use_container_width=True)
        
        st.markdown("---")
        
        # Detailed cluster statistics
        st.subheader("📋 Detailed Cluster Statistics")
        
        # Age Summary
        st.markdown("#### Age Summary by Cluster")
        age_summary = clustered_df.groupby('Cluster')['Age'].describe()
        st.dataframe(age_summary.style.format("{:.2f}"), use_container_width=True)
        
        st.markdown("---")
        
        # Gender and Emotion Distribution
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Gender Distribution by Cluster")
            gender_dist = clustered_df.groupby('Cluster')['Gender'].value_counts(normalize=True).reset_index()
            gender_dist.columns = ['Cluster', 'Gender', 'Proportion']
            gender_dist['Percentage'] = (gender_dist['Proportion'] * 100).round(2)
            
            # Pivot for better display
            gender_pivot = gender_dist.pivot(index='Cluster', columns='Gender', values='Percentage').fillna(0)
            st.dataframe(gender_pivot.style.format("{:.2f}%").background_gradient(cmap='Blues', axis=1), 
                        use_container_width=True)
        
        with col2:
            st.markdown("#### Top Emotions by Cluster")
            emotion_dist = clustered_df.groupby('Cluster')['Dominant_Emotion'].value_counts(normalize=True).reset_index()
            emotion_dist.columns = ['Cluster', 'Emotion', 'Proportion']
            emotion_dist['Percentage'] = (emotion_dist['Proportion'] * 100).round(2)
            
            # Show top 3 emotions per cluster
            top_emotions = emotion_dist.groupby('Cluster').head(3)
            top_emotions_pivot = top_emotions.pivot(index='Cluster', columns='Emotion', values='Percentage').fillna(0)
            st.dataframe(top_emotions_pivot.style.format("{:.2f}%").background_gradient(cmap='RdYlGn', axis=1),
                        use_container_width=True)
        
        st.markdown("---")
        
        # Full emotion distribution table
        st.markdown("#### Complete Emotion Distribution by Cluster (%)")
        emotion_full = clustered_df.groupby('Cluster')['Dominant_Emotion'].value_counts(normalize=True).unstack(fill_value=0) * 100
        st.dataframe(emotion_full.style.format("{:.2f}%").background_gradient(cmap='YlOrRd', axis=None),
                    use_container_width=True)
        
        # Key insights from the distributions
        st.success("""
        **🔍 Key Findings:**
        - **Cluster 0 (Green):** Dominated by **Happiness (72.3%)** - predominantly Female and Male users
        - **Cluster 1 (Blue):** Mixed emotions with **Neutral (28.8%)** and **Boredom (27.0%)** - highest Non-binary representation (45.6%)
        - **Cluster 2 (Red):** Led by **Anger (30.3%)** with diverse emotional mix - balanced gender distribution
        """)
        
        st.markdown("---")
        
        # Cluster engagement patterns
        st.subheader("Average Engagement Metrics by Cluster")
        
        cluster_engagement = clustered_df.groupby('Cluster')[['Posts_Per_Day', 'Likes_Received_Per_Day',
                                                               'Comments_Received_Per_Day', 
                                                               'Messages_Sent_Per_Day']].mean()
        
        # Create interactive plotly chart with cluster colors
        fig = go.Figure()
        metrics = ['Posts_Per_Day', 'Likes_Received_Per_Day', 'Comments_Received_Per_Day', 'Messages_Sent_Per_Day']
        metric_names = ['Posts/Day', 'Likes/Day', 'Comments/Day', 'Messages/Day']
        
        for idx, (metric, name) in enumerate(zip(metrics, metric_names)):
            fig.add_trace(go.Bar(
                x=[f'Cluster {i}' for i in cluster_engagement.index],
                y=cluster_engagement[metric],
                name=name,
                marker_color=[cluster_colors[i] for i in cluster_engagement.index] if idx == 0 else None,
                offsetgroup=idx
            ))
        
        fig.update_layout(
            title='Average Social Media Engagement by Cluster',
            xaxis_title='Cluster',
            yaxis_title='Average Count',
            barmode='group',
            height=500,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
          
        # Pairplot of engagement features
        st.subheader("Pairplot: Engagement Features by Cluster")
        
        st.info("""
        This pairplot visualizes the relationships between all engagement metrics (Posts, Likes, Comments, Messages) 
        colored by cluster. It helps identify patterns and correlations within each cluster.
        """)
        
        # Prepare data for pairplot
        df_pairplot = clustered_df[['Posts_Per_Day', 'Likes_Received_Per_Day', 
                                     'Comments_Received_Per_Day', 'Messages_Sent_Per_Day', 'Cluster']].copy()
        df_pairplot = df_pairplot.rename(columns={
            'Posts_Per_Day': 'Posts',
            'Likes_Received_Per_Day': 'Likes',
            'Comments_Received_Per_Day': 'Comments',
            'Messages_Sent_Per_Day': 'Messages'
        })
        
        # Create pairplot with consistent cluster colors
        cluster_palette = {0: cluster_colors[0], 1: cluster_colors[1], 2: cluster_colors[2]}
        fig = sns.pairplot(df_pairplot, vars=['Posts', 'Likes', 'Comments', 'Messages'], 
                          hue='Cluster', palette=cluster_palette, diag_kind='kde', 
                          plot_kws={'alpha': 0.6, 's': 30})
        fig.fig.suptitle("Pairplot of Social Media Engagement Features by Cluster", y=1.01, fontsize=16)
        
        # Update legend to show cluster names
        for ax in fig.axes.flatten():
            if ax.get_legend():
                handles, labels = ax.get_legend_handles_labels()
                ax.legend(handles, [f'Cluster {l}' for l in labels], 
                         title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        st.pyplot(fig)
        
        st.success("""
        **Key Observations:**
        - Diagonal plots show the distribution of each metric within clusters
        - Off-diagonal plots reveal correlations between different engagement metrics
        - Different cluster colors help identify distinct behavioral patterns
        - Strong positive correlations between Posts, Likes, and Comments are visible
        """)
    
    # TAB 6: STATISTICAL INSIGHTS
    with tab6:
        st.header("📊 Statistical Insights & Correlations")
        
        
        
        # Key statistical findings
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Usage Time vs Likes Correlation")
            
            emotions = ['Happiness', 'Anxiety', 'Anger', 'Sadness', 'Neutral', 'Boredom']
            usage_avgs = []
            likes_avgs = []
            
            for emotion in emotions:
                users = train_df[train_df['Dominant_Emotion'] == emotion]
                usage_avgs.append(users['Daily_Usage_Time (minutes)'].mean())
                likes_avgs.append(users['Likes_Received_Per_Day'].mean())
            
            correlation, p_value = pearsonr(usage_avgs, likes_avgs)
            
            st.metric("Pearson Correlation", f"{correlation:.3f}")
            st.metric("P-value", f"{p_value:.6f}")
            
            if p_value < 0.001:
                st.success("✅ **Highly significant correlation!** (p < 0.001)")
            
            st.info("""
            **Interpretation:**  
            There is a near-perfect positive correlation (0.996) between average 
            usage time and average likes received across emotions. This suggests 
            that increased platform engagement strongly correlates with receiving 
            more social validation.
            """)
        
        with col2:
            st.subheader("Scatter Plot: Usage vs Likes by Emotion")
            
            fig = go.Figure()
            
            colors_map = {
                'Happiness': '#ec96ce',
                'Anxiety': '#b2f134',
                'Anger': '#e15e2c',
                'Sadness': '#1E90FF',
                'Neutral': '#f1f592',
                'Boredom': '#c063e6'
            }
            
            for i, emotion in enumerate(emotions):
                fig.add_trace(go.Scatter(
                    x=[usage_avgs[i]],
                    y=[likes_avgs[i]],
                    mode='markers+text',
                    name=emotion,
                    text=[emotion],
                    textposition="top center",
                    marker=dict(size=15, color=colors_map[emotion]),
                    showlegend=True
                ))
            
            fig.update_layout(
                title="Average Usage Time vs Likes Received by Emotion",
                xaxis_title="Average Daily Usage Time (minutes)",
                yaxis_title="Average Likes Received Per Day",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Summary statistics
        st.subheader("Summary Statistics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### Usage Time")
            st.write(train_df['Daily_Usage_Time (minutes)'].describe())
        
        with col2:
            st.markdown("### Posts Per Day")
            st.write(train_df['Posts_Per_Day'].describe())
        
        with col3:
            st.markdown("### Likes Per Day")
            st.write(train_df['Likes_Received_Per_Day'].describe())
        
        st.markdown("---")
        
        # Key findings summary
        st.subheader("🎯 Key Research Findings")
        
        st.markdown("""
        ### Main Insights from the Analysis:
        
        1. **Emotion-Usage Relationship**
           - Users reporting **Happiness** showed significantly higher platform usage (avg. usage time)
           - Happy users also received more likes and posted more frequently
           - This suggests positive reinforcement mechanisms (dopamine-driven engagement)
        
        2. **Age and Usage Correlation**
           - **Weak correlation** (0.087) between age and daily usage time
           - P-value of 0.38 indicates no statistically significant relationship
           - Age is not a strong predictor of social media usage in this dataset
        
        3. **Platform Preferences**
           - Different emotions show distinct platform distribution patterns
           - Platform choice appears to be influenced by user demographics and emotional states
        
        4. **Engagement Patterns**
           - **Near-perfect correlation (0.996)** between usage time and likes received
           - Users with higher engagement receive more social validation
           - Happiness correlates with both higher usage and higher engagement
        
        5. **Clustering Results**
           - Users naturally segment into 3 distinct behavioral clusters
           - Clusters show different patterns in age, gender, emotion, and engagement levels
           - Cluster 2 shows predominance of happiness; Cluster 3 shows more sadness
        
        6. **Gender Differences**
           - Gender shows varying emotional distributions on social media
           - No dramatic differences in overall usage time across genders
        
        ### Implications:
        
        - **Positive emotional states** are associated with higher platform engagement
        - **Social validation** (likes, comments) may reinforce continued usage
        - Platform design that promotes positive interactions may increase user retention
        - Different user segments require tailored content and engagement strategies
        """)

else:
    st.error("⚠️ Unable to load data. Please check the file path and try again.")
    st.info("Expected file location: `C:/Users/Joanna Gutierrez/Downloads/proj-data-science/train.csv`")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; padding: 20px;'>
    <p>Social Media & Emotions Analysis Dashboard | Data Science Project</p>
    <p>Built with Streamlit 🎈 | © 2025</p>
</div>
""", unsafe_allow_html=True)

