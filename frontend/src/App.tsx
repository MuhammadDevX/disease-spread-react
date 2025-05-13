// Requirements: npm install react-force-graph react-force-graph-2d
import React, { useState, useMemo } from 'react';
import { BrowserRouter as Router, Routes, Route, Link, useLocation } from 'react-router-dom';
import AppBar from '@mui/material/AppBar';
import Box from '@mui/material/Box';
import Toolbar from '@mui/material/Toolbar';
import TabsMUI from '@mui/material/Tabs';
import TabMUI from '@mui/material/Tab';
import Typography from '@mui/material/Typography';
import ScienceIcon from '@mui/icons-material/Science';
import GroupWorkIcon from '@mui/icons-material/GroupWork';
import PolicyIcon from '@mui/icons-material/Policy';
import CompareArrowsIcon from '@mui/icons-material/CompareArrows';
import TextField from '@mui/material/TextField';
import MenuItem from '@mui/material/MenuItem';
import Button from '@mui/material/Button';
import CircularProgress from '@mui/material/CircularProgress';
import Alert from '@mui/material/Alert';
import axios from 'axios';
import Plot from 'react-plotly.js';
import Card from '@mui/material/Card';
import CardContent from '@mui/material/CardContent';
import Grid from '@mui/material/Grid';
import IconButton from '@mui/material/IconButton';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import PauseIcon from '@mui/icons-material/Pause';
import Slider from '@mui/material/Slider';
import ForceGraph2D from 'react-force-graph-2d';

// Add this type declaration if not present in the project
// declare module 'react-force-graph-2d';

function useRouteMatch(patterns: readonly string[]) {
  const { pathname } = useLocation();
  for (let i = 0; i < patterns.length; i += 1) {
    if (pathname === patterns[i]) {
      return patterns[i];
    }
  }
  return false;
}

const defaultParams = {
  num_nodes: 500,
  network_model: 'watts_strogatz',
  model_type: 'SIR',
  transmission_rate: 0.5,
  recovery_rate: 0.1,
  mortality_rate: 0.01,
  latency_period: 5,
  immunity_duration: 100,
  healthcare_capacity: 100,
  initial_infection_rate: 0.01,
  steps: 100,
};

// Helper to map node state to color
const stateColors: Record<string, string> = {
  S: 'blue',
  E: 'yellow',
  I: 'red',
  R: 'green',
  D: 'black',
};

// Memoized network visualization using react-force-graph
const NetworkForceGraph = React.memo(function NetworkForceGraph({
  nodes, edges, nodeStates, highlightInfected, animate, animFrame, infectionHistory
}: {
  nodes: any[],
  edges: any[],
  nodeStates: Record<number, string>,
  highlightInfected?: boolean,
  animate?: boolean,
  animFrame?: number,
  infectionHistory?: Record<number, string>[]
}) {
  // Prepare graph data
  const graphData = useMemo(() => ({
    nodes: nodes.map((n) => ({
      id: n.id,
      state: nodeStates[n.id],
    })),
    links: edges.map((e) => ({
      source: e.source,
      target: e.target,
    })),
  }), [nodes, edges, nodeStates]);

  // For animation, update node states per frame
  const displayStates = animate && infectionHistory && typeof animFrame === 'number'
    ? infectionHistory[animFrame] || nodeStates
    : nodeStates;

  return (
    <ForceGraph2D
      graphData={graphData}
      nodeLabel={(n: any) => `Node ${n.id}\nState: ${displayStates[n.id]}`}
      nodeAutoColorBy={null}
      nodeCanvasObject={(node: any, ctx: CanvasRenderingContext2D, globalScale: number) => {
        const label = `${node.id}`;
        ctx.beginPath();
        ctx.arc(node.x, node.y, displayStates[node.id] === 'I' ? 9 : 6, 0, 2 * Math.PI, false);
        ctx.fillStyle = stateColors[displayStates[node.id]] || 'gray';
        ctx.globalAlpha = displayStates[node.id] === 'D' ? 0.5 : 1;
        ctx.fill();
        ctx.globalAlpha = 1;
        ctx.lineWidth = 2;
        ctx.strokeStyle = 'DarkSlateGrey';
        ctx.stroke();
        ctx.font = `${10 / globalScale}px Sans-Serif`;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillStyle = '#222';
        ctx.fillText(label, node.x, node.y - 12);
      }}
      linkColor={(link: any) => {
        if (!highlightInfected) return 'rgba(128,128,128,0.2)';
        const srcState = displayStates[link.source.id ?? link.source];
        const tgtState = displayStates[link.target.id ?? link.target];
        return (srcState === 'I' || tgtState === 'I') ? 'rgba(255,0,0,0.4)' : 'rgba(128,128,128,0.2)';
      }}
      linkWidth={(link: any) => {
        if (!highlightInfected) return 1;
        const srcState = displayStates[link.source.id ?? link.source];
        const tgtState = displayStates[link.target.id ?? link.target];
        return (srcState === 'I' || tgtState === 'I') ? 2 : 1;
      }}
      width={window.innerWidth * 0.9}
      height={500}
      backgroundColor="#fafcff"
      enableNodeDrag={true}
    />
  );
});

function MainSimulation() {
  const [params, setParams] = useState(defaultParams);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<any>(null);
  const [tab, setTab] = useState('time');
  const [simState, setSimState] = useState<any>(null);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setParams((prev) => ({ ...prev, [name]: isNaN(Number(value)) ? value : Number(value) }));
  };

  const handleSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    setParams((prev) => ({ ...prev, [e.target.name]: e.target.value }));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const response = await axios.post('http://localhost:8000/simulate', params);
      setResult(response.data);
      setSimState(response.data.sim_state || null);
    } catch (err: any) {
      setError(err.message || 'Simulation failed');
    } finally {
      setLoading(false);
    }
  };

  const runSteps = async (numSteps: number) => {
    setLoading(true);
    setError(null);
    try {
      let payload: any = { params: { ...params, steps: numSteps } };
      if (simState) {
        payload.sim_state = simState;
      }
      const response = await axios.post('http://localhost:8000/simulate', payload);
      setResult(response.data);
      setSimState(response.data.sim_state || null);
    } catch (err: any) {
      setError(err.message || 'Simulation failed');
    } finally {
      setLoading(false);
    }
  };

  // Build nodeStates for static network view
  const nodeStates = useMemo(() => {
    if (!result || !result.nodes) return {};
    const map: Record<number, string> = {};
    result.nodes.forEach((n: any) => { map[n.id] = n.state; });
    return map;
  }, [result?.nodes]);

  // --- Advanced Visualizations ---
  const renderNetworkView = () => {
    if (!result || !result.nodes || !result.edges) return null;
    return (
      <NetworkForceGraph
        nodes={result.nodes}
        edges={result.edges}
        nodeStates={nodeStates}
        highlightInfected={true}
      />
    );
  };

  const renderHeatmap = () => {
    // Dummy heatmap for now
    return (
      <Plot
        data={[
          {
            z: [
              [1, 20, 30],
              [20, 1, 60],
              [30, 60, 1],
            ],
            type: 'heatmap',
            colorscale: 'Reds',
          },
        ]}
        layout={{
          title: 'Infection Density Heatmap (Dummy)',
          autosize: true,
        }}
        useResizeHandler
        style={{ width: '100%', height: '400px' }}
        config={{ responsive: true }}
      />
    );
  };

  return (
    <Box p={3}>
      <Typography variant="h4" gutterBottom>Main Simulation</Typography>
      <form onSubmit={handleSubmit} style={{ maxWidth: 600 }}>
        <Box display="flex" flexWrap="wrap" gap={2}>
          <TextField label="Population Size" name="num_nodes" type="number" value={params.num_nodes} onChange={handleChange} fullWidth required />
          <TextField select label="Network Model" name="network_model" value={params.network_model} onChange={handleSelect} fullWidth>
            <MenuItem value="watts_strogatz">Watts-Strogatz</MenuItem>
            <MenuItem value="erdos_renyi">Erdős-Rényi</MenuItem>
            <MenuItem value="barabasi_albert">Barabási-Albert</MenuItem>
          </TextField>
          <TextField select label="Model Type" name="model_type" value={params.model_type} onChange={handleSelect} fullWidth>
            <MenuItem value="SIR">SIR</MenuItem>
            <MenuItem value="SEIR">SEIR</MenuItem>
            <MenuItem value="SIRD">SIRD</MenuItem>
          </TextField>
          <TextField label="Transmission Rate (β)" name="transmission_rate" type="number" value={params.transmission_rate} onChange={handleChange} inputProps={{ step: 0.01, min: 0, max: 1 }} fullWidth required />
          <TextField label="Recovery Rate (γ)" name="recovery_rate" type="number" value={params.recovery_rate} onChange={handleChange} inputProps={{ step: 0.01, min: 0, max: 1 }} fullWidth required />
          <TextField label="Mortality Rate (δ)" name="mortality_rate" type="number" value={params.mortality_rate} onChange={handleChange} inputProps={{ step: 0.005, min: 0, max: 1 }} fullWidth required />
          <TextField label="Latency Period (λ)" name="latency_period" type="number" value={params.latency_period} onChange={handleChange} inputProps={{ step: 1, min: 1, max: 14 }} fullWidth required />
          <TextField label="Immunity Duration" name="immunity_duration" type="number" value={params.immunity_duration} onChange={handleChange} inputProps={{ step: 1, min: 10, max: 365 }} fullWidth required />
          <TextField label="Healthcare Capacity" name="healthcare_capacity" type="number" value={params.healthcare_capacity} onChange={handleChange} inputProps={{ step: 10, min: 10, max: 500 }} fullWidth required />
          <TextField label="Initial Infection Rate" name="initial_infection_rate" type="number" value={params.initial_infection_rate} onChange={handleChange} inputProps={{ step: 0.001, min: 0, max: 0.1 }} fullWidth required />
        </Box>
        <Box mt={2} display="flex" gap={2}>
          <Button variant="outlined" color="secondary" disabled={loading} onClick={() => runSteps(1)}>
            Run One Step
          </Button>
          <Button variant="outlined" color="secondary" disabled={loading} onClick={() => runSteps(10)}>
            Run Ten Steps
          </Button>
        </Box>
      </form>
      {error && <Box mt={2}><Alert severity="error">{error}</Alert></Box>}
      {/* Tabs for advanced visualizations */}
      {result && (
        <Box mt={4}>
          <TabsMUI value={tab} onChange={(_, v) => setTab(v)}>
            <TabMUI label="Time Series" value="time" />
            <TabMUI label="Network View" value="network" />
            <TabMUI label="Heatmap" value="heatmap" />
          </TabsMUI>
          <Box mt={2}>
            {tab === 'time' && (
              <Plot
                data={['S', 'E', 'I', 'R', 'D'].map((state) => ({
                  x: result.time_series.map((_: any, i: number) => i),
                  y: result.time_series.map((step: any) => step[state] || 0),
                  type: 'scatter',
                  mode: 'lines+markers',
                  name: state,
                }))}
                layout={{
                  title: 'Population State Over Time',
                  xaxis: { title: 'Time Step' },
                  yaxis: { title: 'Population Count' },
                  legend: { orientation: 'h' },
                  autosize: true,
                }}
                useResizeHandler
                style={{ width: '100%', height: '400px' }}
                config={{ responsive: true }}
              />
            )}
            {tab === 'network' && renderNetworkView()}
            {tab === 'heatmap' && renderHeatmap()}
          </Box>
        </Box>
      )}
    </Box>
  );
}

function BatchAnalysis() {
  const [params, setParams] = useState({
    ...defaultParams,
    num_runs: 10,
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<any>(null);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setParams((prev) => ({ ...prev, [name]: isNaN(Number(value)) ? value : Number(value) }));
  };
  const handleSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    setParams((prev) => ({ ...prev, [e.target.name]: e.target.value }));
  };
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const response = await axios.post('http://localhost:8000/batch', params);
      setResult(response.data);
    } catch (err: any) {
      setError(err.message || 'Batch analysis failed');
    } finally {
      setLoading(false);
    }
  };
  return (
    <Box p={3}>
      <Typography variant="h4" gutterBottom>Batch Analysis</Typography>
      <form onSubmit={handleSubmit} style={{ maxWidth: 600 }}>
        <Box display="flex" flexWrap="wrap" gap={2}>
          <TextField label="Population Size" name="num_nodes" type="number" value={params.num_nodes} onChange={handleChange} fullWidth required />
          <TextField select label="Network Model" name="network_model" value={params.network_model} onChange={handleSelect} fullWidth>
            <MenuItem value="watts_strogatz">Watts-Strogatz</MenuItem>
            <MenuItem value="erdos_renyi">Erdős-Rényi</MenuItem>
            <MenuItem value="barabasi_albert">Barabási-Albert</MenuItem>
          </TextField>
          <TextField select label="Model Type" name="model_type" value={params.model_type} onChange={handleSelect} fullWidth>
            <MenuItem value="SIR">SIR</MenuItem>
            <MenuItem value="SEIR">SEIR</MenuItem>
            <MenuItem value="SIRD">SIRD</MenuItem>
          </TextField>
          <TextField label="Transmission Rate (β)" name="transmission_rate" type="number" value={params.transmission_rate} onChange={handleChange} inputProps={{ step: 0.01, min: 0, max: 1 }} fullWidth required />
          <TextField label="Recovery Rate (γ)" name="recovery_rate" type="number" value={params.recovery_rate} onChange={handleChange} inputProps={{ step: 0.01, min: 0, max: 1 }} fullWidth required />
          <TextField label="Mortality Rate (δ)" name="mortality_rate" type="number" value={params.mortality_rate} onChange={handleChange} inputProps={{ step: 0.005, min: 0, max: 1 }} fullWidth required />
          <TextField label="Steps" name="steps" type="number" value={params.steps} onChange={handleChange} inputProps={{ step: 1, min: 10, max: 1000 }} fullWidth required />
          <TextField label="Number of Runs" name="num_runs" type="number" value={params.num_runs} onChange={handleChange} inputProps={{ step: 1, min: 1, max: 100 }} fullWidth required />
        </Box>
        <Box mt={2}>
          <Button type="submit" variant="contained" color="primary" disabled={loading}>
            {loading ? <CircularProgress size={24} /> : 'Run Batch Analysis'}
          </Button>
        </Box>
      </form>
      {error && <Box mt={2}><Alert severity="error">{error}</Alert></Box>}
      {result && (
        <Box mt={4}>
          <Typography variant="h6">Batch Results (Mean Peak Infected: {result.summary?.mean_peak_infected?.toFixed(2)})</Typography>
          {/* --- Statistics Cards --- */}
          <Grid container spacing={2} mb={2}>
            {result.statistics && Object.keys(result.statistics).map((metric) => (
              <Grid key={metric} size={{ xs: 12, sm: 6, md: 4, lg: 3 }}>
                <Card variant="outlined" sx={{ minWidth: 200, background: '#f5faff' }}>
                  <CardContent>
                    <Typography variant="subtitle1" color="primary" gutterBottom>{metric.replace(/_/g, ' ').toUpperCase()}</Typography>
                    <Typography variant="body2">Mean: <b>{result.statistics[metric].mean.toFixed(2)}</b></Typography>
                    <Typography variant="body2">Median: <b>{result.statistics[metric].median.toFixed(2)}</b></Typography>
                    <Typography variant="body2">Std: <b>{result.statistics[metric].std.toFixed(2)}</b></Typography>
                    <Typography variant="body2">Min: <b>{result.statistics[metric].min.toFixed(2)}</b></Typography>
                    <Typography variant="body2">Max: <b>{result.statistics[metric].max.toFixed(2)}</b></Typography>
                  </CardContent>
                </Card>
              </Grid>
            ))}
          </Grid>
          {/* --- Box Plots for Key Metrics --- */}
          <Grid container spacing={2} mb={2}>
            {result.statistics && Object.keys(result.statistics).map((metric) => (
              <Grid key={metric + '-box'} size={{ xs: 12, sm: 6, md: 4, lg: 3 }}>
                <Plot
                  data={[{
                    y: result.statistics[metric].all_values,
                    type: 'box',
                    name: metric.replace(/_/g, ' ').toUpperCase(),
                    boxpoints: 'all',
                    jitter: 0.5,
                    marker: { color: '#1976d2' },
                  }]}
                  layout={{
                    title: metric.replace(/_/g, ' ').toUpperCase() + ' Distribution',
                    autosize: true,
                    margin: { t: 40, l: 40, r: 20, b: 40 },
                  }}
                  useResizeHandler
                  style={{ width: '100%', height: '300px' }}
                  config={{ responsive: true }}
                />
              </Grid>
            ))}
          </Grid>
          {/* --- Example Run Plot --- */}
          <Plot
            data={result.runs[0] ? ['S', 'E', 'I', 'R', 'D'].map((state) => ({
              x: result.runs[0].time_series.map((_: any, i: number) => i),
              y: result.runs[0].time_series.map((step: any) => step[state] || 0),
              type: 'scatter',
              mode: 'lines+markers',
              name: state,
            })) : []}
            layout={{
              title: 'Example Run (First Simulation Time Series)',
              xaxis: { title: 'Time Step' },
              yaxis: { title: 'Population Count' },
              legend: { orientation: 'h' },
              autosize: true,
            }}
            useResizeHandler
            style={{ width: '100%', height: '400px' }}
            config={{ responsive: true }}
          />
        </Box>
      )}
    </Box>
  );
}

function PolicyInterventions() {
  const [params, setParams] = useState(defaultParams);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<any>(null);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setParams((prev) => ({ ...prev, [name]: isNaN(Number(value)) ? value : Number(value) }));
  };
  const handleSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    setParams((prev) => ({ ...prev, [e.target.name]: e.target.value }));
  };
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const response = await axios.post('http://localhost:8000/policy', params);
      setResult(response.data);
    } catch (err: any) {
      setError(err.message || 'Policy analysis failed');
    } finally {
      setLoading(false);
    }
  };
  return (
    <Box p={3}>
      <Typography variant="h4" gutterBottom>Policy Interventions</Typography>
      <form onSubmit={handleSubmit} style={{ maxWidth: 600 }}>
        <Box display="flex" flexWrap="wrap" gap={2}>
          <TextField label="Population Size" name="num_nodes" type="number" value={params.num_nodes} onChange={handleChange} fullWidth required />
          <TextField select label="Network Model" name="network_model" value={params.network_model} onChange={handleSelect} fullWidth>
            <MenuItem value="watts_strogatz">Watts-Strogatz</MenuItem>
            <MenuItem value="erdos_renyi">Erdős-Rényi</MenuItem>
            <MenuItem value="barabasi_albert">Barabási-Albert</MenuItem>
          </TextField>
          <TextField select label="Model Type" name="model_type" value={params.model_type} onChange={handleSelect} fullWidth>
            <MenuItem value="SIR">SIR</MenuItem>
            <MenuItem value="SEIR">SEIR</MenuItem>
            <MenuItem value="SIRD">SIRD</MenuItem>
          </TextField>
          <TextField label="Transmission Rate (β)" name="transmission_rate" type="number" value={params.transmission_rate} onChange={handleChange} inputProps={{ step: 0.01, min: 0, max: 1 }} fullWidth required />
          <TextField label="Recovery Rate (γ)" name="recovery_rate" type="number" value={params.recovery_rate} onChange={handleChange} inputProps={{ step: 0.01, min: 0, max: 1 }} fullWidth required />
          <TextField label="Mortality Rate (δ)" name="mortality_rate" type="number" value={params.mortality_rate} onChange={handleChange} inputProps={{ step: 0.005, min: 0, max: 1 }} fullWidth required />
          <TextField label="Steps" name="steps" type="number" value={params.steps} onChange={handleChange} inputProps={{ step: 1, min: 10, max: 1000 }} fullWidth required />
        </Box>
        <Box mt={2}>
          <Button type="submit" variant="contained" color="primary" disabled={loading}>
            {loading ? <CircularProgress size={24} /> : 'Run Policy Analysis'}
          </Button>
        </Box>
      </form>
      {error && <Box mt={2}><Alert severity="error">{error}</Alert></Box>}
      {result && (
        <Box mt={4}>
          <Typography variant="h6">With Policy (Time Series)</Typography>
          <Plot
            data={['S', 'E', 'I', 'R', 'D'].map((state) => ({
              x: result.with_policy.time_series.map((_: any, i: number) => i),
              y: result.with_policy.time_series.map((step: any) => step[state] || 0),
              type: 'scatter',
              mode: 'lines+markers',
              name: state,
            }))}
            layout={{
              title: 'With Policy',
              xaxis: { title: 'Time Step' },
              yaxis: { title: 'Population Count' },
              legend: { orientation: 'h' },
              autosize: true,
            }}
            useResizeHandler
            style={{ width: '100%', height: '300px' }}
            config={{ responsive: true }}
          />
          <Typography variant="h6" mt={4}>Without Policy (Time Series)</Typography>
          <Plot
            data={['S', 'E', 'I', 'R', 'D'].map((state) => ({
              x: result.without_policy.time_series.map((_: any, i: number) => i),
              y: result.without_policy.time_series.map((step: any) => step[state] || 0),
              type: 'scatter',
              mode: 'lines+markers',
              name: state,
            }))}
            layout={{
              title: 'Without Policy',
              xaxis: { title: 'Time Step' },
              yaxis: { title: 'Population Count' },
              legend: { orientation: 'h' },
              autosize: true,
            }}
            useResizeHandler
            style={{ width: '100%', height: '300px' }}
            config={{ responsive: true }}
          />
        </Box>
      )}
    </Box>
  );
}

function ModelComparison() {
  const [params, setParams] = useState(defaultParams);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<any>(null);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setParams((prev) => ({ ...prev, [name]: isNaN(Number(value)) ? value : Number(value) }));
  };
  const handleSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    setParams((prev) => ({ ...prev, [e.target.name]: e.target.value }));
  };
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const response = await axios.post('http://localhost:8000/compare', params);
      setResult(response.data);
    } catch (err: any) {
      setError(err.message || 'Model comparison failed');
    } finally {
      setLoading(false);
    }
  };
  return (
    <Box p={3}>
      <Typography variant="h4" gutterBottom>Model Comparison</Typography>
      <form onSubmit={handleSubmit} style={{ maxWidth: 600 }}>
        <Box display="flex" flexWrap="wrap" gap={2}>
          <TextField label="Population Size" name="num_nodes" type="number" value={params.num_nodes} onChange={handleChange} fullWidth required />
          <TextField select label="Network Model" name="network_model" value={params.network_model} onChange={handleSelect} fullWidth>
            <MenuItem value="watts_strogatz">Watts-Strogatz</MenuItem>
            <MenuItem value="erdos_renyi">Erdős-Rényi</MenuItem>
            <MenuItem value="barabasi_albert">Barabási-Albert</MenuItem>
          </TextField>
          <TextField label="Transmission Rate (β)" name="transmission_rate" type="number" value={params.transmission_rate} onChange={handleChange} inputProps={{ step: 0.01, min: 0, max: 1 }} fullWidth required />
          <TextField label="Recovery Rate (γ)" name="recovery_rate" type="number" value={params.recovery_rate} onChange={handleChange} inputProps={{ step: 0.01, min: 0, max: 1 }} fullWidth required />
          <TextField label="Mortality Rate (δ)" name="mortality_rate" type="number" value={params.mortality_rate} onChange={handleChange} inputProps={{ step: 0.005, min: 0, max: 1 }} fullWidth required />
          <TextField label="Latency Period (λ)" name="latency_period" type="number" value={params.latency_period} onChange={handleChange} inputProps={{ step: 1, min: 1, max: 14 }} fullWidth required />
          <TextField label="Immunity Duration" name="immunity_duration" type="number" value={params.immunity_duration} onChange={handleChange} inputProps={{ step: 1, min: 10, max: 365 }} fullWidth required />
          <TextField label="Healthcare Capacity" name="healthcare_capacity" type="number" value={params.healthcare_capacity} onChange={handleChange} inputProps={{ step: 10, min: 10, max: 500 }} fullWidth required />
          <TextField label="Initial Infection Rate" name="initial_infection_rate" type="number" value={params.initial_infection_rate} onChange={handleChange} inputProps={{ step: 0.001, min: 0, max: 0.1 }} fullWidth required />
          <TextField label="Steps" name="steps" type="number" value={params.steps} onChange={handleChange} inputProps={{ step: 1, min: 10, max: 1000 }} fullWidth required />
        </Box>
        <Box mt={2}>
          <Button type="submit" variant="contained" color="primary" disabled={loading}>
            {loading ? <CircularProgress size={24} /> : 'Run Model Comparison'}
          </Button>
        </Box>
      </form>
      {error && <Box mt={2}><Alert severity="error">{error}</Alert></Box>}
      {result && (
        <Box mt={4}>
          <Typography variant="h6">Model Comparison Results</Typography>
          <Plot
            data={Object.keys(result.results).map((modelType) => ({
              x: result.results[modelType].time_series.map((_: any, i: number) => i),
              y: result.results[modelType].time_series.map((step: any) => step['I'] || 0),
              type: 'scatter',
              mode: 'lines+markers',
              name: modelType + ' (Infected)',
            }))}
            layout={{
              title: 'Infected Over Time (All Models)',
              xaxis: { title: 'Time Step' },
              yaxis: { title: 'Population Count' },
              legend: { orientation: 'h' },
              autosize: true,
            }}
            useResizeHandler
            style={{ width: '100%', height: '400px' }}
            config={{ responsive: true }}
          />
        </Box>
      )}
    </Box>
  );
}

function NavTabs() {
  const routeMatch = useRouteMatch([
    '/',
    '/batch',
    '/policy',
    '/compare',
  ]);
  const currentTab = routeMatch ? routeMatch : false;
  return (
    <TabsMUI value={currentTab} textColor="inherit" indicatorColor="secondary">
      <TabMUI icon={<ScienceIcon />} label="Main Simulation" value="/" component={Link} to="/" />
      <TabMUI icon={<GroupWorkIcon />} label="Batch Analysis" value="/batch" component={Link} to="/batch" />
      <TabMUI icon={<PolicyIcon />} label="Policy Interventions" value="/policy" component={Link} to="/policy" />
      <TabMUI icon={<CompareArrowsIcon />} label="Model Comparison" value="/compare" component={Link} to="/compare" />
    </TabsMUI>
  );
}

function App() {
  return (
    <Router>
      <Box sx={{ flexGrow: 1 }}>
        <AppBar position="static" color="primary">
          <Toolbar>
            <Typography variant="h6" sx={{ flexGrow: 1 }}>
              Disease Spread Simulation
            </Typography>
            <NavTabs />
          </Toolbar>
        </AppBar>
        <Routes>
          <Route path="/" element={<MainSimulation />} />
          <Route path="/batch" element={<BatchAnalysis />} />
          <Route path="/policy" element={<PolicyInterventions />} />
          <Route path="/compare" element={<ModelComparison />} />
        </Routes>
      </Box>
    </Router>
  );
}

export default App;
