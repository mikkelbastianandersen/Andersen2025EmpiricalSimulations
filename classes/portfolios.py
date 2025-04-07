# portfolio.py
class Portfolio:
    def __init__(self, assets):
        self.holdings = {asset.name: 0 for asset in assets}

    def update_holdings(self, new_allocations):
        self.holdings.update(new_allocations)

    def calculate_value(self, returns):
        return sum(weight * returns[name] for name, weight in self.holdings.items())

    def esg_impact(self, esg_scores):
        return sum(weight * esg_scores[name] for name, weight in self.holdings.items())
