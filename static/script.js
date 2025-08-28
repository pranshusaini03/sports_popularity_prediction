document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('predictionForm');
    const loadingDiv = document.getElementById('loading');
    const errorDiv = document.getElementById('error');
    const predictionResult = document.getElementById('predictionResult');
    const plotDiv = document.getElementById('plot');
    const accuracyPlot = document.getElementById('accuracyPlot');
    const checkAccuracyBtn = document.getElementById('checkAccuracy');
    const predictionTable = document.getElementById('predictionTable').getElementsByTagName('tbody')[0];

    let currentData = null;

    form.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        // Get form data
        const sport = document.getElementById('sport').value;
        const model = document.getElementById('model').value;
        
        // Validate input
        if (!sport || !model) {
            errorDiv.style.display = 'block';
            errorDiv.querySelector('p').textContent = 'Please select both sport and model';
            return;
        }

        try {
            // Show loading state
            loadingDiv.style.display = 'block';
            errorDiv.style.display = 'none';
            predictionResult.style.display = 'none';
            accuracyPlot.style.display = 'none';

            // Make prediction request
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    sport: sport,
                    model_type: model
                })
            });

            const data = await response.json();

            if (response.ok) {
                currentData = data;
                
                // Create visualization
                const trace = {
                    x: data.dates,
                    y: data.predictions,
                    type: 'scatter',
                    name: 'Predictions',
                    line: {
                        color: '#3498db'
                    }
                };

                const layout = {
                    title: `${sport.charAt(0).toUpperCase() + sport.slice(1)} Popularity Prediction`,
                    xaxis: {
                        title: 'Date'
                    },
                    yaxis: {
                        title: 'Popularity Index'
                    },
                    hovermode: 'x unified',
                    showlegend: true
                };

                Plotly.newPlot(plotDiv, [trace], layout);
                
                // Update prediction table
                predictionTable.innerHTML = '';
                data.dates.forEach((date, index) => {
                    const row = predictionTable.insertRow();
                    const dateCell = row.insertCell(0);
                    const predictionCell = row.insertCell(1);
                    dateCell.textContent = date;
                    predictionCell.textContent = data.predictions[index].toFixed(2);
                });

                // Show results
                predictionResult.style.display = 'block';
            } else {
                throw new Error(data.error || 'Prediction failed');
            }
        } catch (error) {
            errorDiv.style.display = 'block';
            errorDiv.querySelector('p').textContent = `Error: ${error.message}`;
        } finally {
            loadingDiv.style.display = 'none';
        }
    });

    checkAccuracyBtn.addEventListener('click', async function() {
        if (!currentData) return;

        try {
            // Show loading state
            loadingDiv.style.display = 'block';
            errorDiv.style.display = 'none';

            // Make accuracy check request
            const response = await fetch('/check_accuracy', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    sport: document.getElementById('sport').value,
                    model_type: document.getElementById('model').value
                })
            });

            const data = await response.json();

            if (response.ok) {
                // Create accuracy visualization
                const trace1 = {
                    x: data.historical_dates,
                    y: data.historical_values,
                    type: 'scatter',
                    name: 'Historical Data',
                    line: {
                        color: '#2ecc71'
                    }
                };

                const trace2 = {
                    x: data.prediction_dates,
                    y: data.prediction_values,
                    type: 'scatter',
                    name: 'Predictions',
                    line: {
                        color: '#e74c3c',
                        dash: 'dot'
                    }
                };

                const layout = {
                    title: 'Model Accuracy Analysis',
                    xaxis: {
                        title: 'Date'
                    },
                    yaxis: {
                        title: 'Popularity Index'
                    },
                    hovermode: 'x unified',
                    showlegend: true
                };

                Plotly.newPlot(accuracyPlot, [trace1, trace2], layout);
                accuracyPlot.style.display = 'block';
            } else {
                throw new Error(data.error || 'Failed to check accuracy');
            }
        } catch (error) {
            errorDiv.style.display = 'block';
            errorDiv.querySelector('p').textContent = `Error: ${error.message}`;
        } finally {
            loadingDiv.style.display = 'none';
        }
    });
}); 