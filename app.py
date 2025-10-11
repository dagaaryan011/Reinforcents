# app.py
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import json
import time
from datetime import datetime
from config import DATA_FILE_PATH # Import the raw string path

class LiveTradingDashboard:
    def __init__(self):
        # Uses the correct path
        self.data_file = DATA_FILE_PATH
        self.setup_page()

    def load_data(self):
        try:
            # Reads from the correct path
            with open(self.data_file, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return None
    def setup_page(self):
        st.set_page_config(
            page_title="Reinforcents🪙",
            page_icon="",
            layout="wide"
        )
        st.title("NESTLE - (NESTLEIND.NS)")

    

    def create_price_chart(self, data):
        timestamps = [datetime.fromisoformat(ts) for ts in data['price_timestamps']]
        fig = go.Figure(go.Scatter(
            x=timestamps,
            y=data['price_values'],
            mode='lines',
            line=dict(color='#00D4AA', width=2),
            name='Stock Price'
        ))
        fig.update_layout(
            title="Price",
            template="plotly_dark",
            height=400,
            xaxis=dict(rangeslider=dict(visible=True), type="date"),
            yaxis_title="Price (₹)"
        )
        return fig

    def create_portfolio_chart(self, portfolio_data, title, color):
        timestamps = [datetime.fromisoformat(ts) for ts in portfolio_data['timestamps']]
        fig = go.Figure(go.Scatter(
            x=timestamps,
            y=portfolio_data['values'],
            mode='lines',
            line=dict(color=color, width=2),
            name=title
        ))
        fig.update_layout(
            template="plotly_dark",
            height=250,
            margin=dict(l=20, r=20, t=40, b=20),
            yaxis_title="Value (₹)"
        )
        return fig

    def display_leaderboard(self, leaderboard_data, title):
        st.markdown(f"**{title}**")
        if not leaderboard_data:
            st.info("No data available.")
            return
        
        df = pd.DataFrame(leaderboard_data)
        df = df.rename(columns={
            "name": "Agent",
            "portfolio_value": "Portfolio Value",
            "profit_pct": "Profit %"
        })
        # Add Rank
        df.index = df.index + 1
        df.index.name = "Rank"

        st.dataframe(df[['Agent', 'Portfolio Value', 'Profit %']],
            column_config={
                "Portfolio Value": st.column_config.NumberColumn(format="₹%.0f"),
                "Profit %": st.column_config.NumberColumn(format="%.1f%%")
            }
        )

    def run(self):
        data = self.load_data()

        if not data or not data.get('price_timestamps'):
            st.warning("⏳ Waiting for simulation data... Start main.py to begin!")
            time.sleep(2)
            st.rerun()

        # Main price chart
        st.plotly_chart(self.create_price_chart(data), use_container_width=True)
        st.markdown("---")

        # Top Performers Section
        st.subheader(" Best Performing Agents")
        cols = st.columns(3)
        
        agent_types = [
            ('best_insti_portfolio', '🏛️ Institution', '#FF6B6B'),
            ('best_mm_portfolio', '💼 Market Maker', '#4ECDC4'),
            ('best_retail_portfolio', '👥 Retail', '#45B7D1')
        ]

        for i, (key, name, color) in enumerate(agent_types):
            with cols[i]:
                agent_data = data[key]
                if agent_data['agent_id']:
                    st.metric(
                        f"{name}: {agent_data['agent_id']}",
                        f"₹{agent_data['current_value']:,.0f}",
                        f"{agent_data['profit_pct']:+.1f}%"
                    )
                    st.plotly_chart(self.create_portfolio_chart(agent_data, name, color), use_container_width=True)
                else:
                    st.info(f"Waiting for {name.split(' ')[1]} data...")
        
        st.markdown("---")

        # Leaderboards Section
        st.subheader(" Agent Leaderboards")
        lb_cols = st.columns(3)
        
        with lb_cols[0]:
            self.display_leaderboard(data['insti_leaderboard'], "🏛️ Institution Leaderboard")
        with lb_cols[1]:
            self.display_leaderboard(data['mm_leaderboard'], "💼 Market Maker Leaderboard")
        with lb_cols[2]:
            self.display_leaderboard(data['retail_leaderboard'], "👥 Retail Leaderboard")

        # Sidebar Status
        st.sidebar.title("📈 Simulation Status")
        last_update = datetime.fromisoformat(data['last_update'])
        st.sidebar.metric("Last Update", last_update.strftime('%H:%M:%S'))
        st.sidebar.metric("Current Price", f"₹{data['price_values'][-1]:.2f}")
        st.sidebar.metric("Data Points", len(data['price_values']))

        # Auto-refresh
        time.sleep(2)
        st.rerun()

if __name__ == "__main__":
    dashboard = LiveTradingDashboard()
    dashboard.run()