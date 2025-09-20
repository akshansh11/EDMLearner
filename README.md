# EDMLearner - Intelligent EDM Process Optimization

A comprehensive Streamlit-based application for optimizing Electrical Discharge Machining (EDM) parameters using machine learning techniques and interactive visualizations. This platform focuses on Ti-13Zr-13Nb titanium alloy machining for orthopedic implant manufacturing.

## Features

### Core Modules

- **Data Explorer**: Interactive correlation analysis and dataset overview
- **Process Optimizer**: Multi-objective optimization with Pareto frontier analysis
- **ML Predictions**: Machine learning models vs. empirical predictions
- **3D Surface Analysis**: Interactive response surface methodology visualization
- **Digital Twin Simulator**: Real-time process simulation and monitoring

### Key Capabilities

- **Empirical Model Integration**: Implementation of validated mathematical models from research
- **Machine Learning Models**: Random Forest and Gradient Boosting regressors
- **Interactive Visualizations**: 3D surface plots, contour maps, and correlation heatmaps
- **Multi-Objective Optimization**: Balance material removal rate, electrode wear, and surface quality
- **Real-time Parameter Adjustment**: Live prediction updates with parameter changes
- **Process Health Monitoring**: Automated recommendations and status indicators

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/akshansh11/EDMLearner.git
cd EDMLearner
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Add the EDM process image:
   - Place `EDM.png` in the project root directory
   - The image will be displayed in the app header

4. Run the application:
```bash
streamlit run app.py
```

## Dataset

The application uses experimental data from a face-centered Central Composite Design study with:

- **30 experimental runs**
- **Input Parameters**: Voltage (50-70V), Current (8-16A), Pulse ON time (6-10μs), Pulse OFF time (7-11μs)
- **Output Responses**: Material Removal Rate (MRR), Electrode Wear Rate (EWR), Surface Roughness (SR)
- **Material**: Ti-13Zr-13Nb alloy specimens
- **Process**: 20-minute EDM machining with graphite electrodes

## Usage

### Data Explorer
- Examine correlations between process parameters
- View dataset statistics and experimental design
- Understand parameter relationships

### Process Optimizer
- Set optimization objectives and weights
- Define process constraints
- Identify optimal parameter combinations
- Visualize Pareto frontier solutions

### ML Predictions
- Compare empirical models with machine learning approaches
- Perform sensitivity analysis
- Evaluate model performance metrics
- Make real-time predictions

### 3D Surface Analysis
- Generate interactive response surfaces
- Create contour plots for any parameter combination
- Visualize complex parameter interactions

### Digital Twin Simulator
- Real-time process monitoring
- Parameter adjustment with live feedback
- Process efficiency calculations
- Automated recommendations

## Technical Details

### Models Implemented

1. **Empirical Models**: Original mathematical models from the research paper
2. **Random Forest**: Ensemble learning for robust predictions
3. **Gradient Boosting**: Advanced ensemble method for complex relationships

### Optimization Methods

- Multi-objective optimization using weighted composite scores
- Pareto frontier analysis for trade-off identification
- Response Surface Methodology (RSM) implementation

### Visualization Technologies

- **Plotly**: Interactive 3D surface plots and contour maps
- **Seaborn/Matplotlib**: Statistical visualizations
- **Streamlit**: Web interface and dashboard components

## File Structure

```
EDMLearner/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── EDM.png               # Process illustration image
├── README.md             # This file
└── data/                 # Optional: additional datasets
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Data Source and Citation

This application is based on experimental data from:

**Soundhar, A., Zubar, H.A., Sultan, M.T.B.H.H. and Kandasamy, J., 2019. Dataset on optimization of EDM machining parameters by using central composite design. *Data in brief*, *23*, p.103671.**

DOI: [10.1016/j.dib.2019.01.019](https://doi.org/10.1016/j.dib.2019.01.019)

The original research was conducted at:
- School of Mechanical Engineering, VIT University, Vellore, Tamil Nadu, India
- Faculty of Engineering, King Abdulaziz University, Jeddah, Saudi Arabia
- Laboratory of Biocomposite Technology, Universiti Putra Malaysia

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Original research team for providing comprehensive experimental data
- VIT University for materials science research
- Open access publication enabling data replication and analysis
- Streamlit community for visualization frameworks

## Applications

This platform is designed for:

- **Manufacturing Engineers**: Process optimization and quality control
- **Research Scientists**: EDM parameter analysis and validation
- **Students**: Learning EDM principles and optimization techniques
- **Medical Device Manufacturers**: Orthopedic implant production optimization

## Support

For questions or issues:
- Open an issue on GitHub
- Review the documentation in the app's help sections
- Check the original research paper for detailed methodology

---

**Keywords**: Electrical Discharge Machining, Titanium Alloys, Machine Learning, Process Optimization, Orthopedic Implants, Response Surface Methodology
