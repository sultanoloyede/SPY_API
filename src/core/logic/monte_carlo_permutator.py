import random
from typing import List
from dataclasses import replace
from src.core.adapters.yf_market_adapter import YFMarketDataAdapter
from src.core.models.bar import Bar


class MonteCarloPermutator:

    def __init__(self, adapter: YFMarketDataAdapter, num_permutations: int = 100):
        self.adapter = adapter
        self.num_permutations = num_permutations
        self._permuted_adapters: List[YFMarketDataAdapter] = []
        self._generate_permutations()

    def _generate_permutations(self):
        original_data = self.adapter._list_data
        adapter_class = type(self.adapter)

        # Extract all timestamps from original data, sorted 
        # TODO: Sample weekdays and weekends independtly to get rid of large jumps in data
        ordered_timestamps = [bar.timestamp for bar in original_data]


        for _ in range(self.num_permutations):
            
            previous_bar = original_data[0]
            permuted_data: list[Bar] = []
            for idx in range(len(original_data)):

                rdm_idx = random.randint(1, len(original_data)-1)
                bar_smpl = original_data[rdm_idx]
                prev_bar_smpl = original_data[rdm_idx-1]

                # Open relative to last close
                new_open = (bar_smpl.open - prev_bar_smpl.close) + previous_bar.close 
                new_close = (bar_smpl.close - bar_smpl.open) + new_open
                
                # Compute potential new high and low
                potential_high = (bar_smpl.high - bar_smpl.open) + new_open
                potential_low = (bar_smpl.low - bar_smpl.open) + new_open
                
                # Check that HL respect the invariance
                new_low = min(new_open, new_close, potential_low)
                new_high = max(new_open, new_close, potential_high)

                previous_bar = Bar(
                    timestamp=ordered_timestamps[idx],
                    open=new_open,
                    close=new_close,
                    high=new_high,
                    low=new_low
                )
                permuted_data.append(previous_bar)
            
            adapter_instance = adapter_class(self.adapter.asset)
            adapter_instance._list_data = permuted_data # TODO: Refactor code to handle setting historical data externally
            self._permuted_adapters.append(adapter_instance)

    @property
    def permuted_adapters(self) -> List[YFMarketDataAdapter]:
        return self._permuted_adapters
