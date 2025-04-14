from src.generate_network import generate_social_network
from src.visualize_network import visualize_social_network
from src.simulate_disease import simulate_disease_spread
from src.generate_report import generate_summary

def main():
    G = generate_social_network(num_nodes=500, model='watts_strogatz')
    visualize_social_network(G, "Initial State")

    G = simulate_disease_spread(G, p_infection=0.08, steps=15)
    visualize_social_network(G, "After Simulation")

    report_text = "This simulation modeled the spread of disease across a town of 500 people. An initial 5 infections spread over 15 time steps with a probability of infection set to 0.08."
    summary = generate_summary(report_text)
    print("Simulation Summary:\n", summary)

if __name__ == "__main__":
    main()
