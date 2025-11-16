import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#TODO 

# Improve the view of the indices
def compute_contributions(best_particles, swarm_size, max_iteration):
    """
    Compute how many times each particle was the best overall and per defined period.

    Parameters
    ----------
    best_particles : list or np.array
        best_particles[t] = index of best particle at iteration t (0-based or 1-based)
    swarm_size : int
        Number of particles in the swarm.
    decades : list[int]
        Key iteration thresholds defining periods, e.g., [10, 20, 50, 100, 1000]

    Returns
    -------
    contrib_df : pd.DataFrame
        DataFrame where rows = particles (1..swarm_size)
        columns = each period + Total
    """

    dates = [i*10 for i in range(max_iteration//10 +1)]
    # Ensure numpy array
    best_particles = np.array(best_particles)
    particle_indices = np.arange(1, swarm_size + 1)
    display_indices = particle_indices

    # Define period boundaries
    periods = [[dates[i],dates[i+1]] for i in range(len(dates)-1)]
    n_periods = len(periods)

    # Initialize contributions array
    contributions = np.zeros((swarm_size, n_periods), dtype=int)

    Iperiod = 0
    period = periods[Iperiod]
    for it,bestId in enumerate(best_particles):
        print(it)
        # Identify the current period
        while  not(min(period)<=it and it < max(period)): 
            Iperiod +=1
            period = periods[Iperiod]
        
        contributions[bestId - 1 , Iperiod] += 1 

    # Build DataFrame
    period_labels = [f"{min(periods[i])}-{max(periods[i])}" for i in range(len(periods))]
    contrib_df = pd.DataFrame(contributions, columns=period_labels, index=display_indices)
    contrib_df["Total"] = contrib_df.sum(axis=1)
    contrib_df.index.name = "Particle"
 
    return dates, contrib_df


def plot_contributions(contrib_df, decades):
    """
    Plot stacked bar chart of particle contributions.

    Parameters
    ----------
    contrib_df : pd.DataFrame
        Output from compute_contributions()
    decades : list[int]
        The iteration breakpoints
    """
    particles = contrib_df.index
    period_labels = [col for col in contrib_df.columns if col != "Total"]

    fig, ax = plt.subplots(figsize=(10, 6))
    bottom = np.zeros(len(particles))

    # Stacked bar per period
    for period in period_labels:
        values = contrib_df[period].values
        p = ax.bar(particles, values, label=period, bottom=bottom)
        bottom += values
    plt.xticks([i for i in particles])
    ax.set_xlabel("Particle index")
    ax.set_ylabel("Number of times best")
    ax.set_title("Particle Contributions per Period | swarmsize = {}".format(len(particles)))
    ax.legend(title="Iteration Periods")
    plt.tight_layout()
  


def save_contributions_to_excel(contrib_df, best_particles, filepath):
    """
    Save contributions summary and full best particle history to one Excel file.

    Parameters
    ----------
    contrib_df : pd.DataFrame
        Contribution summary table.
    best_particles : list or np.array
        The full list of best particles over iterations.
    filepath : str
        Output Excel file path.
    """
    with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
        contrib_df.to_excel(writer, sheet_name="Contributions")
        pd.DataFrame({"Iteration": np.arange(len(best_particles)),
                      "BestParticle": best_particles}).to_excel(writer, sheet_name="BestHistory", index=False)
    print(f"✅ Results saved to {filepath}")

if __name__ == "__main__":

    max_iteration = 20
    swarmsize = 5
    import random
    random.seed(42)

    best = [random.randint(1,swarmsize) for _ in range(19)]
    random.shuffle(best)
    print(best)
    decades, compt = compute_contributions(best, swarmsize, max_iteration)
    plot_contributions(compt, decades)
    print(decades, compt)

    pass