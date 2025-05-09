from simulation import run_simulation
import pandas as pd

collector = run_simulation()

returns_df = pd.DataFrame(collector.returns)
returns_df.to_excel(
    r"C:\Users\mia\OneDrive - Danske Commodities\Desktop\TestReturns.xlsx", index=False
)
