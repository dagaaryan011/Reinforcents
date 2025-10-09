import json
import time
from datetime import datetime
import threading
import os

class DataManager:
    def __init__(self, data_file="simulation_data.json"):
        self.data_file = data_file
        self.lock = threading.Lock()
        # Track starting values for accurate profit calculation
        self.starting_values = {}
        self.initialize_data()
    
    def initialize_data(self):
        """Initialize empty data structure"""
        initial_data = {
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
        self.save_data(initial_data)
    
    def save_data(self, data):
        """Save data to JSON file with error handling"""
        with self.lock:
            try:
                # Clean the data to ensure JSON serializability
                cleaned_data = self.clean_data(data)
                cleaned_data['last_update'] = datetime.now().isoformat()
                
                # Write to a temporary file first, then rename (atomic operation)
                temp_file = self.data_file + '.tmp'
                with open(temp_file, 'w') as f:
                    json.dump(cleaned_data, f, indent=2, default=str)
                
                # Atomic replace
                if os.path.exists(self.data_file):
                    os.replace(temp_file, self.data_file)
                else:
                    os.rename(temp_file, self.data_file)
                    
            except Exception as e:
                # Silent error handling - no console output
                try:
                    backup_file = self.data_file + '.backup'
                    with open(backup_file, 'w') as f:
                        json.dump({'error': str(e), 'timestamp': datetime.now().isoformat()}, f)
                except:
                    pass
    
    def clean_data(self, data):
        """Clean data to ensure it's JSON serializable"""
        if isinstance(data, dict):
            return {k: self.clean_data(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self.clean_data(item) for item in data]
        elif isinstance(data, (int, float)):
            # Handle NaN and infinity
            if data != data:  # NaN
                return 0
            elif data == float('inf'):
                return 1e10
            elif data == float('-inf'):
                return -1e10
            else:
                return float(data)  # Convert to float for consistency
        elif isinstance(data, (str, bool, type(None))):
            return data
        else:
            return str(data)  # Convert any other type to string
    
    def load_data(self):
        """Load data from JSON file with robust error handling"""
        try:
            with self.lock:
                if not os.path.exists(self.data_file):
                    self.initialize_data()
                    return self.load_data()
                
                with open(self.data_file, 'r') as f:
                    content = f.read().strip()
                    
                if not content:
                    self.initialize_data()
                    return self.load_data()
                
                return json.loads(content)
                
        except (json.JSONDecodeError, KeyError, ValueError):
            # Silent recovery - no console output
            try:
                if os.path.exists(self.data_file):
                    backup_file = self.data_file + '.corrupted'
                    os.rename(self.data_file, backup_file)
            except:
                pass
            
            self.initialize_data()
            return self.load_data()
    
    def update_live_data(self, new_price, insti_envs, mm_agents, retail_envs):
        """Update data from simulation with CORRECT profit calculation"""
        data = self.load_data()
        current_time = datetime.now().isoformat()
        
        # Update price path
        data['price_timestamps'].append(current_time)
        data['price_values'].append(float(new_price))
        
        # Process INSTITUTIONAL agents
        insti_leaderboard = []
        for env in insti_envs:
            try:
                agent_id = env.agent.agent_id
                
                # Store starting value if not already stored
                start_key = f"insti_{agent_id}"
                if start_key not in self.starting_values:
                    self.starting_values[start_key] = float(env.portfolio_value)
                
                starting_value = self.starting_values[start_key]
                current_value = float(env.portfolio_value)
                
                # Calculate ACTUAL profit percentage
                if starting_value > 0:
                    profit_pct = ((current_value - starting_value) / starting_value) * 100
                else:
                    profit_pct = 0
                
                # Get trade count
                trade_count = getattr(env.agent, 'trade_count', 0)
                if isinstance(trade_count, float):
                    trade_count = int(trade_count)
                
                insti_leaderboard.append({
                    'name': str(agent_id),
                    'profit_pct': float(profit_pct),
                    'portfolio_value': float(current_value),
                    'starting_value': float(starting_value),
                    'trades': trade_count
                })
            except Exception:
                continue
        
        if insti_leaderboard:
            best_insti = max(insti_leaderboard, key=lambda x: x['portfolio_value'])
            data['best_insti_portfolio']['timestamps'].append(current_time)
            data['best_insti_portfolio']['values'].append(float(best_insti['portfolio_value']))
            data['best_insti_portfolio']['current_value'] = float(best_insti['portfolio_value'])
            data['best_insti_portfolio']['profit_pct'] = float(best_insti['profit_pct'])
            data['best_insti_portfolio']['agent_id'] = str(best_insti['name'])
        
        # Process MARKET MAKER agents
        mm_leaderboard = []
        for agent in mm_agents:
            try:
                agent_id = agent.agent_id
                
                # Store starting value if not already stored
                start_key = f"mm_{agent_id}"
                if start_key not in self.starting_values:
                    self.starting_values[start_key] = float(agent.broker.portfolio_value)
                
                starting_value = self.starting_values[start_key]
                current_value = float(agent.broker.portfolio_value)
                
                # Calculate ACTUAL profit percentage
                if starting_value > 0:
                    profit_pct = ((current_value - starting_value) / starting_value) * 100
                else:
                    profit_pct = 0
                
                # Get trade count
                trade_count = getattr(agent, 'trade_count', 0)
                if isinstance(trade_count, float):
                    trade_count = int(trade_count)
                
                mm_leaderboard.append({
                    'name': str(agent_id),
                    'profit_pct': float(profit_pct),
                    'portfolio_value': float(current_value),
                    'starting_value': float(starting_value),
                    'trades': trade_count
                })
            except Exception:
                continue
        
        if mm_leaderboard:
            best_mm = max(mm_leaderboard, key=lambda x: x['portfolio_value'])
            data['best_mm_portfolio']['timestamps'].append(current_time)
            data['best_mm_portfolio']['values'].append(float(best_mm['portfolio_value']))
            data['best_mm_portfolio']['current_value'] = float(best_mm['portfolio_value'])
            data['best_mm_portfolio']['profit_pct'] = float(best_mm['profit_pct'])
            data['best_mm_portfolio']['agent_id'] = str(best_mm['name'])
        
        # Process RETAIL agents
        retail_leaderboard = []
        for env in retail_envs:
            try:
                agent_id = env.agent.agent_id
                
                # Calculate current portfolio value
                positions_value = sum(float(quantity) * float(new_price) for quantity in env.portfolio.values())
                current_value = float(env.cash_balance) + positions_value
                
                # Store starting value if not already stored
                start_key = f"retail_{agent_id}"
                if start_key not in self.starting_values:
                    self.starting_values[start_key] = current_value
                
                starting_value = self.starting_values[start_key]
                
                # Calculate ACTUAL profit percentage
                if starting_value > 0:
                    profit_pct = ((current_value - starting_value) / starting_value) * 100
                else:
                    profit_pct = 0
                
                # Get trade count
                trade_count = getattr(env.agent, 'trade_count', 0)
                if isinstance(trade_count, float):
                    trade_count = int(trade_count)
                
                retail_leaderboard.append({
                    'name': str(agent_id),
                    'profit_pct': float(profit_pct),
                    'portfolio_value': float(current_value),
                    'starting_value': float(starting_value),
                    'trades': trade_count
                })
            except Exception:
                continue
        
        if retail_leaderboard:
            best_retail = max(retail_leaderboard, key=lambda x: x['portfolio_value'])
            data['best_retail_portfolio']['timestamps'].append(current_time)
            data['best_retail_portfolio']['values'].append(float(best_retail['portfolio_value']))
            data['best_retail_portfolio']['current_value'] = float(best_retail['portfolio_value'])
            data['best_retail_portfolio']['profit_pct'] = float(best_retail['profit_pct'])
            data['best_retail_portfolio']['agent_id'] = str(best_retail['name'])
        
        # Store ALL leaderboards
        data['insti_leaderboard'] = sorted(insti_leaderboard, key=lambda x: x['profit_pct'], reverse=True)[:10]
        data['mm_leaderboard'] = sorted(mm_leaderboard, key=lambda x: x['profit_pct'], reverse=True)[:10]
        data['retail_leaderboard'] = sorted(retail_leaderboard, key=lambda x: x['profit_pct'], reverse=True)[:10]
        
        self.save_data(data)