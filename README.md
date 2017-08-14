# Inductive Loop Visualizations
Interactive Dash visualization of inductive loop data from the Porto Living Lab.
Work done as part of collaboration with Instituto de Telecomunicações, University of Porto, Portugal, which owns all data used for this visualization.

This project uses Python 3, though both Python 2 and 3 are supported by Dash.

To set up Dash, install the following in your Python environment (see https://plot.ly/dash/installation):

`pip install dash==0.17.7  # The core dash backend`

`pip install dash-renderer==0.7.4  # The dash front-end`

`pip install dash-html-components==0.7.0  # HTML components`

`pip install dash-core-components==0.12.0  # Supercharged components`

`pip install plotly==2.0.13  # Plotly graphing library used in examples`

Additionally, `pandas`, `numpy`, `calendar`, `dateutil`, `datetime` and `ast` are required.

To specify server and port on which to run the app, modify line 435 in loopCorrelations.py as below:
`app.run_server(host='HOST_IP', port=PORT_NO, debug=True)`
