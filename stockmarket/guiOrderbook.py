import numpy as np
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path\


# Set the page to always be in wide mode
st.set_page_config(layout="wide")

# Assuming these are in your local directory
from orderbook import OrderBook, Order, Side 
from noise import generate_samples

# --- CSV setup for persistent storage ---
# This check ensures the file is created on first run if it doesn't exist.
ORDERS_FILE = Path("orders.csv")
if not ORDERS_FILE.exists():
    orders_df = pd.DataFrame(columns=["ID", "Side", "Price", "Size"])
    orders_df.to_csv(ORDERS_FILE, index=False)

# --- Init session state ---
if "ob" not in st.session_state:
    st.session_state.ob = OrderBook(market_price=100.0)

if "samples" not in st.session_state:
    st.session_state.samples, st.session_state.open_price, st.session_state.close_price, \
    st.session_state.high_price, st.session_state.low_price = generate_samples()

ob = st.session_state.ob
samples = st.session_state.samples
open_price = st.session_state.open_price
close_price = st.session_state.close_price
high_price = st.session_state.high_price
low_price = st.session_state.low_price

# --- Title ---
st.title("📈 Reliance")

col1, col2 = st.columns([65, 35])

# LEFT → Stagnant Graph
# The graph is now plotted once using all sample data
with col1:
  
    st.subheader("📊 Market Depth Chart")

    # 1. Get the current bid and ask data from the order book
    bids = ob.get_bids()
    asks = ob.get_asks()

    if not bids and not asks:
        st.write("Order book is empty. No depth to display.")
    else:
        # 2. Prepare the data for plotting
        bid_prices, bid_sizes = zip(*bids) if bids else ([], [])
        ask_prices, ask_sizes = zip(*asks) if asks else ([], [])

        # 3. Calculate cumulative sizes for the depth chart
        cumulative_bid_sizes = np.cumsum(bid_sizes)
        cumulative_ask_sizes = np.cumsum(ask_sizes)

        # 4. Plot the data using Matplotlib
        fig, ax = plt.subplots()
        ax.grid(True, axis='y', linestyle='--', alpha=0.6)

        # Plot the bid and ask steps
        # ax.step(bid_prices, cumulative_bid_sizes, where='pre', color='green', lw=1.5, label='Bids (Buy Orders)')
        # ax.step(ask_prices, cumulative_ask_sizes, where='pre', color='red', lw=1.5, label='Asks (Sell Orders)')

        # Fill the area under the steps to create the depth effect
        ax.fill_between(bid_prices, cumulative_bid_sizes, step='pre', color='green', alpha=0.2)
        ax.fill_between(ask_prices, cumulative_ask_sizes, step='pre', color='red', alpha=0.2)

        # 5. Format the chart to look clean and professional
        ax.set_title("Order Book Depth")
        ax.set_xlabel("Price (₹)")
        ax.set_ylabel("Cumulative Size")
        ax.legend()
        
        # Smartly set the x-axis limits to focus on the spread
        if bids and asks:
            spread = ask_prices[0] - bid_prices[0]
            center = bid_prices[0] + spread / 2
            ax.set_xlim(center - spread* 2, center + spread*2 )
        
        # Invert the x-axis to have bids on the left and asks on the right
        ax.invert_xaxis()
        
        st.pyplot(fig)
        plt.close(fig)
   


# RIGHT → Order book
# This section will automatically update on a form submission
with col2:
    st.subheader("📑 Order Book")

    buy_col, sell_col = st.columns(2)

    with buy_col:
        st.markdown("### 🟢 Buy Orders")
        bids = ob.get_bids()
        if bids:
            bids_df = pd.DataFrame(bids, columns=["Price", "Size"])
            # Set the index to be 1-based
            bids_df.index = range(1, len(bids_df) + 1)
            st.table(bids_df.style.format({'Price': '{:.2f}'}))
        else:
            st.write("No BUY orders")

    with sell_col:
        st.markdown("### 🔴 Sell Orders")
        asks = ob.get_asks()
        if asks:
            asks_df = pd.DataFrame(asks, columns=["Price", "Size"])
            # Set the index to be 1-based
            asks_df.index = range(1, len(asks_df) + 1)
            st.table(asks_df.style.format({'Price': '{:.2f}'}))
        else:
            st.write("No SELL orders")

    st.markdown("### ⚡ Executed Trades")
    # The get_trades() method now handles reading the CSV file automatically.
    trades = ob.get_trades()
    if trades:
        trades_df = pd.DataFrame(trades, columns=["Side", "Price", "Size"])
        # Set the index to be 1-based
        trades_df.index = range(1, len(trades_df) + 1)
        # st.table(trades_df)
        st.table(trades_df.style.format({'Price': '{:.2f}'}))
    else:
        st.write("No trades yet")

# --- Add new orders ---
with st.expander("➕ Add New Order", expanded=False):
    with st.form("order_form"):
        st.markdown("### Enter Order Details")
        # Now using strings "BUY" and "SELL" for a cleaner UI
        side_str = st.selectbox("Side", ["BUY", "SELL"])
        default_price = ob.market_price if ob.market_price else 100.0 
        price = st.number_input("Price", step=0.05, value=default_price, format="%.2f")
        size = st.number_input("Size", step=1)
        submitted = st.form_submit_button("Submit Order")

        if submitted:
            # Convert the string back to the Side enum before creating the order
            side = Side[side_str]

            # Create a simplified order object to pass to the order book
            order = Order(side, price, size)
            
            # Call the new method in orderbook.py
            ob.add_order(order)
            
            # Save the new order string to session state for confirmation message
            st.session_state.last_order = str(order)
            
            st.success(f"Added {order}")

# Show confirmation if last order exists
if "last_order" in st.session_state and st.session_state.last_order:
    st.info(f"Last Order: {st.session_state.last_order}")

# streamlit run D:\Reinforcents\stockmarket\guiOrderbook.py --server.address=172.21.192.1 --server.port=8501