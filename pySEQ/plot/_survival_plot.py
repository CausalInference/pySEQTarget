import matplotlib.pyplot as plt
import polars as pl

def _survival_plot(self):
    if self.plot_type == "risk":
        plot_data = self.km_data.filter(pl.col("estimate") == "risk")
    else: 
        plot_data = self.km_data.filter(pl.col("estimate") == "survival")
    
    for i in self.treatment_level:
        subset = plot_data.filter(pl.col(self.treatment_col) == i)
        plt.plot(subset["followup"], subset["pred"], "-",
                 label = f"treatment = {i}")
    
    plt.xlabel("Followup")
    plt.ylabel(self.plot_type)
    plt.legend()
    plt.grid()
    plt.show()