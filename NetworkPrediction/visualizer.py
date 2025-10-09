import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import json
import time
from datetime import datetime
import os

class LiveTradingDashboard:
    def __init__(self, data_file="simulation_data.json"):
        self.data_file = data_file
        self.setup_page()
    
    def setup_page(self):
        st.set_page_config(
            page_title="Live Options Trading Simulation",
            page_icon="📈",
            layout="wide"
        )
        st.title("🎯 Live Options Trading Simulation")
        st.markdown("---")
    
    def load_data(self):
        """Load data from shared JSON file with robust error handling"""
        try:
            if not os.path.exists(self.data_file):
                return self.get_empty_data()
            
            with open(self.data_file, 'r') as f:
                content = f.read().strip()
                
            if not content:
                return self.get_empty_data()
                
            data = json.loads(content)
            
            # Validate required structure
            if not isinstance(data, dict):
                return self.get_empty_data()
                
            return data
            
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            return self.get_empty_data()
    
    def get_empty_data(self):
        """Return empty data structure"""
        return {
            'price_timestamps': [],
            'price_values': [],
            'best_insti_portfolio': {'timestamps': [], 'values': [], 'current_value': 0, 'profit_pct': 0, 'agent_id': ''},
            'best_mm_portfolio': {'timestamps': [], 'values': [], 'current_value': 0, 'profit_pct': 0, 'agent_id': ''},
            'best_retail_portfolio': {'timestamps': [], 'values': [], 'current_value': 0, 'profit_pct': 0, 'agent_id': ''},
            'insti_leaderboard': [],
            'mm_leaderboard': [],
            'retail_leaderboard': [],
            'last_update': None
        }
    
    def create_price_chart(self, data):
        """Create price chart with FULL history"""
        if not data or not data['price_timestamps']:
            return go.Figure()
        
        # Convert timestamps
        timestamps = [datetime.fromisoformat(ts) for ts in data['price_timestamps']]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=data['price_values'],
            mode='lines',
            line=dict(color='#00D4AA', width=2),
            name='Stock Price'
        ))
        
        fig.update_layout(
            title=dict(
                text="📊 Live Price Path (Full History)",
                font=dict(size=20, color='white')
            ),
            plot_bgcolor='#1e1e1e',
            paper_bgcolor='#1e1e1e',
            font=dict(color='white'),
            margin=dict(l=40, r=40, t=60, b=40),
            height=400,
            xaxis=dict(
                gridcolor='#444444',
                title='Time',
                rangeslider=dict(visible=True),
                type="date"
            ),
            yaxis=dict(
                gridcolor='#444444',
                title='Price ($)'
            )
        )
        
        return fig
    
    def create_portfolio_chart(self, portfolio_data, title, color):
        """Create portfolio value chart for best agents"""
        if not portfolio_data or not portfolio_data['timestamps']:
            return go.Figure()
        
        # Convert timestamps
        timestamps = [datetime.fromisoformat(ts) for ts in portfolio_data['timestamps']]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=portfolio_data['values'],
            mode='lines+markers',
            line=dict(color=color, width=2),
            marker=dict(size=4),
            name=title
        ))
        
        fig.update_layout(
            title=dict(
                text=f"{title} - {portfolio_data.get('agent_id', 'N/A')}",
                font=dict(size=16, color='white')
            ),
            plot_bgcolor='#1e1e1e',
            paper_bgcolor='#1e1e1e',
            font=dict(color='white'),
            margin=dict(l=40, r=40, t=60, b=40),
            height=300,
            xaxis=dict(
                gridcolor='#444444',
                type="date"
            ),
            yaxis=dict(
                gridcolor='#444444',
                title='Portfolio Value ($)'
            )
        )
        
        return fig
    
    def create_leaderboard_table(self, leaderboard_data, title):
        """Create styled leaderboard table with proper trade count handling"""
        if not leaderboard_data:
            st.info(f"No {title} data available")
            return
        
        st.markdown(f"**{title}**")
        
        table_html = f"""
        <div style="background-color: #1e1e1e; padding: 15px; border-radius: 10px; margin: 10px 0;">
            <table style="width: 100%; color: white; border-collapse: collapse; font-size: 14px;">
                <thead>
                    <tr style="border-bottom: 2px solid #00D4AA;">
                        <th style="padding: 10px; text-align: left; font-weight: bold;">Rank</th>
                        <th style="padding: 10px; text-align: left; font-weight: bold;">Agent</th>
                        <th style="padding: 10px; text-align: right; font-weight: bold;">Profit %</th>
                        <th style="padding: 10px; text-align: right; font-weight: bold;">Portfolio Value</th>
                        <th style="padding: 10px; text-align: right; font-weight: bold;">Trades</th>
                    </tr>
                </thead>
                <tbody>
        """
        
        colors = ['#FFD700', '#C0C0C0', '#CD7F32', '#FFFFFF', '#FFFFFF']
        
        for i, agent in enumerate(leaderboard_data[:5]):
            color = colors[i] if i < 3 else colors[3]
            profit_color = '#00ff88' if agent['profit_pct'] >= 0 else '#ff4444'
            
            # Handle trade count - convert to int if it's a float, show as 0 if missing
            trades = agent.get('trades', 0)
            if isinstance(trades, float):
                trades = int(trades)  # Convert float to int
            elif not isinstance(trades, int):
                trades = 0  # Default to 0 if not a number
            
            table_html += f"""
                <tr>
                    <td style="padding: 8px; color: {color}; font-weight: {'bold' if i < 3 else 'normal'}">{i+1}</td>
                    <td style="padding: 8px; color: {color}; font-weight: {'bold' if i < 3 else 'normal'}">{agent['name']}</td>
                    <td style="padding: 8px; text-align: right; color: {profit_color}; font-weight: {'bold' if i < 3 else 'normal'}">{agent['profit_pct']:.1f}%</td>
                    <td style="padding: 8px; text-align: right; color: {color}; font-weight: {'bold' if i < 3 else 'normal'}">${agent['portfolio_value']:,.0f}</td>
                    <td style="padding: 8px; text-align: right; color: {color}; font-weight: {'bold' if i < 3 else 'normal'}">{trades}</td>
                </tr>
            """
        
        table_html += """
                </tbody>
            </table>
        </div>
        """
        
        st.markdown(table_html, unsafe_allow_html=True)
    
    def display_dashboard(self):
        """Display the complete dashboard"""
        data = self.load_data()
        
        if not data or not data.get('price_timestamps'):
            st.warning("⏳ Waiting for simulation data to start...")
            st.info("Start your simulation in another terminal to see live data!")
            time.sleep(2)
            st.rerun()
            return
        
        # Main price chart
        price_chart = self.create_price_chart(data)
        if price_chart.data:
            st.plotly_chart(price_chart, use_container_width=True)
        else:
            st.info("📊 Waiting for price data...")
        
        st.markdown("---")
        
        # Portfolio metrics and charts for ALL THREE categories
        st.subheader("🏆 Best Performing Agents - Portfolio Evolution")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Institution metrics and chart
            insti_data = data['best_insti_portfolio']
            if insti_data.get('current_value', 0) > 0:
                # Calculate actual profit amount for display
                profit_amount = insti_data['current_value'] * (insti_data['profit_pct'] / 100)
                
                st.metric(
                    f"🏛️ {insti_data.get('agent_id', 'Institution')}",
                    f"${insti_data['current_value']:,.0f}",
                    f"{insti_data['profit_pct']:+.1f}% (${profit_amount:,.0f})"
                )
                fig_insti = self.create_portfolio_chart(insti_data, "Best Institution", '#FF6B6B')
                if fig_insti.data:
                    st.plotly_chart(fig_insti, use_container_width=True)
                else:
                    st.info("No institution portfolio history yet")
            else:
                st.info("🏛️ No institution data yet")
        
        with col2:
            # Market Maker metrics and chart
            mm_data = data['best_mm_portfolio']
            if mm_data.get('current_value', 0) > 0:
                profit_amount = mm_data['current_value'] * (mm_data['profit_pct'] / 100)
                
                st.metric(
                    f"💼 {mm_data.get('agent_id', 'Market Maker')}", 
                    f"${mm_data['current_value']:,.0f}",
                    f"{mm_data['profit_pct']:+.1f}% (${profit_amount:,.0f})"
                )
                fig_mm = self.create_portfolio_chart(mm_data, "Best Market Maker", '#4ECDC4')
                if fig_mm.data:
                    st.plotly_chart(fig_mm, use_container_width=True)
                else:
                    st.info("No market maker portfolio history yet")
            else:
                st.info("💼 No market maker data yet")
        
        with col3:
            # Retail metrics and chart
            retail_data = data['best_retail_portfolio']
            if retail_data.get('current_value', 0) > 0:
                profit_amount = retail_data['current_value'] * (retail_data['profit_pct'] / 100)
                
                st.metric(
                    f"👥 {retail_data.get('agent_id', 'Retail')}",
                    f"${retail_data['current_value']:,.0f}",
                    f"{retail_data['profit_pct']:+.1f}% (${profit_amount:,.0f})"
                )
                fig_retail = self.create_portfolio_chart(retail_data, "Best Retail", '#45B7D1')
                if fig_retail.data:
                    st.plotly_chart(fig_retail, use_container_width=True)
                else:
                    st.info("No retail portfolio history yet")
            else:
                st.info("👥 No retail data yet")
        
        st.markdown("---")
        
        # Leaderboards for ALL THREE categories
        st.subheader("📊 Agent Leaderboards (Sorted by Profit %)")
        
        col4, col5, col6 = st.columns(3)
        
        with col4:
            self.create_leaderboard_table(data.get('insti_leaderboard', []), "🏛️ Institution Leaderboard")
        
        with col5:
            self.create_leaderboard_table(data.get('mm_leaderboard', []), "💼 Market Maker Leaderboard")
        
        with col6:
            self.create_leaderboard_table(data.get('retail_leaderboard', []), "👥 Retail Leaderboard")
        
        # Clean simulation info sidebar
        if data.get('last_update'):
            last_update = datetime.fromisoformat(data['last_update'])
            current_price = data['price_values'][-1] if data.get('price_values') else 'N/A'
            current_price_str = f"${current_price:.2f}" if isinstance(current_price, (int, float)) else current_price
            
            st.sidebar.title("📈 Simulation Status")
            st.sidebar.metric("Last Update", last_update.strftime('%H:%M:%S'))
            st.sidebar.metric("Current Price", current_price_str)
            st.sidebar.metric("Data Points", len(data.get('price_values', [])))
        
        # Add manual refresh button
        if st.sidebar.button("🔄 Refresh Now"):
            st.rerun()
        
        # Auto-refresh
        refresh_rate = st.sidebar.selectbox("Auto-refresh", [1, 2, 5, 10], index=1)
        time.sleep(refresh_rate)
        st.rerun()

# Run the dashboard
if __name__ == "__main__":
    dashboard = LiveTradingDashboard()
    dashboard.display_dashboard()