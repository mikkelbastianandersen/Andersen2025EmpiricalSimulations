
class Portfolio:
    def __init__(self, assets):
        self.assets = assets
        self.holdings = {asset.name: 0.0 for asset in assets}

    def update_holdings(self, allocations):
        for i, asset in enumerate(self.assets):
            self.holdings[asset.name] = allocations[i]

    def calculate_return(self, returns):
        return sum(weight * returns[i] for i, weight in enumerate(self.holdings.values()))

    def calculate_esg(self, esg_scores):
        return sum(weight * esg_scores[i] for i, weight in enumerate(self.holdings.values()))