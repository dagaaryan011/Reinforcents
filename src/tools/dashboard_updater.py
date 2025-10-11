# dashboard_updater.py
import json
import os
from datetime import datetime
from config import DATA_FILE_PATH,START_CAPITAL_INSTI,START_CAPITAL_MM,START_CAPITAL_RETAIL

class SimulationState:
    def __init__(self):
        self.data = {
            'price_timestamps': [], 'price_values': [],
            'best_insti_portfolio': self._get_empty_portfolio(),
            'best_mm_portfolio': self._get_empty_portfolio(),
            'best_retail_portfolio': self._get_empty_portfolio(),
            'insti_leaderboard': [], 'mm_leaderboard': [], 'retail_leaderboard': [],
            'last_update': None
        }
    def _get_empty_portfolio(self):
        return {'timestamps': [], 'values': [], 'current_value': 0, 'profit_pct': 0, 'agent_id': ''}
    def update(self, price, insti_portfolios, mm_portfolios, retail_portfolios):
        now_iso = datetime.now().isoformat()
        self.data['price_timestamps'].append(now_iso)
        self.data['price_values'].append(price)

        
        self.data['insti_leaderboard'] = self._create_leaderboard(insti_portfolios, start_capital=START_CAPITAL_INSTI)
        self.data['mm_leaderboard'] = self._create_leaderboard(mm_portfolios, start_capital=START_CAPITAL_MM)
        self.data['retail_leaderboard'] = self._create_leaderboard(retail_portfolios, start_capital=START_CAPITAL_RETAIL)
        

        self._update_best_portfolio('best_insti_portfolio', self.data['insti_leaderboard'], now_iso)
        self._update_best_portfolio('best_mm_portfolio', self.data['mm_leaderboard'], now_iso)
        self._update_best_portfolio('best_retail_portfolio', self.data['retail_leaderboard'], now_iso)
        self.data['last_update'] = now_iso
        self.write_to_file()
    def _create_leaderboard(self, portfolios, start_capital=1_000_000):
        leaderboard = [{'name': agent_id, 'portfolio_value': value, 'profit_pct': ((value - start_capital) / start_capital) * 100 if start_capital > 0 else 0, 'trades': 0} for agent_id, value in portfolios.items()]
        return sorted(leaderboard, key=lambda x: x['profit_pct'], reverse=True)
    def _update_best_portfolio(self, key, leaderboard, timestamp):
        if not leaderboard: return
        best_agent = leaderboard[0]
        self.data[key]['agent_id'] = best_agent['name']
        self.data[key]['current_value'] = best_agent['portfolio_value']
        self.data[key]['profit_pct'] = best_agent['profit_pct']
        self.data[key]['timestamps'].append(timestamp)
        self.data[key]['values'].append(best_agent['portfolio_value'])
    def write_to_file(self):
        try:
            # This now correctly writes to your chosen path
            with open(DATA_FILE_PATH, 'w') as f: 
                json.dump(self.data, f)
        except IOError as e: 
            print(f"Error writing to data file: {e}")
    def clear(self):
        self.__init__()
        self.write_to_file()

state = SimulationState()
def clear_data():
    state.clear()
def update_data(price, insti_portfolios, mm_portfolios, retail_portfolios):
    state.update(price, insti_portfolios, mm_portfolios, retail_portfolios)