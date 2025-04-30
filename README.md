# Advanced Disease Spread Simulation

An interactive web application for simulating disease spread in social networks with various intervention strategies. This project implements advanced epidemiological models and provides tools for analyzing the impact of different network structures and intervention policies.

## Features

- **Multiple Network Models**

  - Watts-Strogatz: Models local clustering and long-distance connections
  - Erdős-Rényi: Provides random network structure
  - Barabási-Albert: Simulates scale-free networks with preferential attachment

- **Advanced Disease Models**

  - SIR (Susceptible-Infected-Recovered)
  - SEIR (Susceptible-Exposed-Infected-Recovered)
  - SIRD (Susceptible-Infected-Recovered-Dead)

- **Intervention Strategies**

  - Vaccination (random and targeted)
  - Lockdown measures
  - Healthcare capacity management
  - Custom policy triggers

- **Visualization Tools**

  - Interactive network visualization
  - Time series plots
  - Infection density heatmaps
  - Comparative analysis charts

- **Batch Simulation**
  - Run multiple simulations in parallel
  - Compare different network models
  - Statistical analysis of results
  - Confidence interval calculations

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/disease-spread-simulation.git
cd disease-spread-simulation
```

2. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

1. Start the Streamlit application:

```bash
streamlit run app.py
```

2. Open your web browser and navigate to the provided URL (typically http://localhost:8501)

3. Configure simulation parameters in the sidebar:

   - Network settings (size, model)
   - Disease parameters (transmission, recovery, mortality rates)
   - Healthcare system capacity
   - Intervention policies

4. Generate a network and run simulations:
   - Use the "Generate New Network" button to create a network
   - Run simulations step by step or in batches
   - Observe the effects of different interventions
   - Analyze results using the provided visualizations

## Project Structure

```
disease-spread-simulation/
├── app.py                  # Main Streamlit application
├── requirements.txt        # Project dependencies
├── README.md              # Project documentation
└── src/
    ├── disease_model.py   # Disease spread simulation logic
    ├── generate_network.py # Network generation and analysis
    ├── policy_engine.py   # Intervention policy management
    ├── visualize_network.py # Visualization tools
    ├── batch_simulation.py # Batch simulation and analysis
    └── batch_interface.py # Batch simulation user interface
```

## Dependencies

- Python 3.8+
- Streamlit
- NetworkX
- NumPy
- Pandas
- Plotly
- SciPy
- python-louvain
- joblib

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- NetworkX for graph theory and network analysis
- Streamlit for the web interface
- Plotly for interactive visualizations
- The epidemiological modeling community for inspiration and guidance
