import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from constants import CM_VEHICLE, CM_READER_MODE
import numpy as np
from lr_model import FuelDistanceLR
from typing import Tuple


def plot_fuel_consumption_and_err(df, vehicle_type):
    df_orig = df.rename(
        columns={'OBFCM Fuel consumption (l/100 km)': 'OBFCM', 'WLTP Fuel consumption (l/100 km)': 'WLTP'}
    )
    df_orig = df_orig.loc[:, ['Manufacturer', 'Fuel Type', 'OBFCM', 'WLTP']]
    df_melted = pd.melt(
        df_orig,
        id_vars=['Manufacturer', 'Fuel Type'],
        value_name='l/100km',
        value_vars=['OBFCM', 'WLTP'],
        var_name='mode',
    )
    # df_cars_aggregated[df_cars_aggregated['Fuel Type'] == 'PETROL'].sort_values(by='value', inplace = True)
    df_melted.sort_values(by=['l/100km'], ascending=False, inplace=True)
    if vehicle_type == 'van':
        category_orders = {'Fuel Type': ['PETROL/ELECTRIC', 'DIESEL', 'PETROL']}
    else:
        category_orders = {'Fuel Type': ['DIESEL/ELECTRIC', 'PETROL/ELECTRIC', 'DIESEL', 'PETROL']}
    fig = px.bar(
        df_melted,
        x='Manufacturer',
        y='l/100km',
        color='mode',
        color_discrete_map=CM_READER_MODE,
        barmode='group',  # barmode shall be as group
        facet_row='Fuel Type',
        # facet_col_wrap=2,
        # text_auto=True,
        text_auto='.2s',
        category_orders=category_orders,
    )
    fig.update_layout(
        bargroupgap=1.0,
        bargap=0.5,  # The greater, the nearer the bars in a group are.
        font_size=8,
        legend=dict(  # Adjust the vertical position of the legend
            traceorder='normal',  # Reverse the order of legend items
            font_size=12,  # Set the font size of legend items
            title='Mode',  # Set the title of the legend
            itemsizing='constant',  # Maintain constant item size
        ),
        title={
            'text': f'Fuel Consumed for {vehicle_type} according to OBFCM and WLPT',
            'y': 0.92,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
        },
    )
    fig.update_traces(
        textfont_size=12,
        textangle=0,
        textposition='outside',
        cliponaxis=False,
        marker=dict(line=dict(width=0.5)),
        width=0.2,
    )  # Set desired bar width
    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=True)
    fig.for_each_annotation(lambda a: a.update(text=a.text.split('=')[-1], font={'size': 7}))
    # fig.update_yaxes(matches=None) # Make the y axes smaller adjustes to the current values.
    fig.write_html(file=f'results/bar_plot_{vehicle_type}.html')
    fig.write_image(f'results/bar_plot_{vehicle_type}.png')
    fig.show()

def plot_gap_consumption(df_car: pd.DataFrame, df_van: pd.DataFrame):
    df_car['vehicle_type'] = 'car'
    df_van['vehicle_type'] = 'van'
    df = pd.concat([df_van, df_car])
    df_orig = df.rename(
        columns={'OBFCM Fuel consumption (l/100 km)': 'OBFCM', 'WLTP Fuel consumption (l/100 km)': 'WLTP'}
    )
    df_orig = df_orig.loc[:, ['Manufacturer', 'vehicle_type', 'Fuel Type', 'OBFCM', 'WLTP']]
    # NOTE: 2nd plot:In thje report is taken as reference WLTP, but the results are strange, therefore, OBFCM is taken which is always greater (lower percentages)
    df_orig['err_consumption'] = df_orig.apply(
        lambda row: round((np.abs(row['OBFCM'] - row['WLTP']) / row['OBFCM']) * 100, 2), axis=1
    )
    df_err_agg = pd.melt(
        df_orig,
        id_vars=['Manufacturer', 'Fuel Type', 'vehicle_type', 'err_consumption'],
        value_vars=['OBFCM', 'WLTP'],
        var_name='mode',
    )

    fig = px.box(
        df_err_agg,
        x='Fuel Type',
        y='err_consumption',
        color_discrete_map=CM_VEHICLE,
        color='vehicle_type',
        category_orders={'Fuel Type': ['DIESEL', 'PETROL', 'PETROL/ELECTRIC', 'DIESEL/ELECTRIC']},
        points='all',
    )
    fig.update_layout(
        title={
            'text': 'Box plot of error (%) between OBFCM and WLTP',
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
        },
        yaxis_title='Error (%)',
        boxgroupgap=0.4,
        boxgap=0.1,  # The greater, the nearer the bars in a group are.
        font_size=8,
        legend=dict(  # Adjust the vertical position of the legend
            traceorder='normal',  # Reverse the order of legend items
            font_size=12,  # Set the font size of legend items
            title='Vehicle Type',  # Set the title of the legend
            itemsizing='constant',  # Maintain constant item size
        ),
    )
    fig.show()
    fig.write_image(f'results/box_error_general.png')

def plot_lifetime_convergence(df_car_raw, df_van_raw):
    def create_line_data_points(x, intercept, coefficient):
        return intercept + x*coefficient

    def trim_data(data) -> Tuple[int, int]:
        Q1 = np.percentile(data, 25)
        Q3 = np.percentile(data, 75)
        IQR = Q3 - Q1
        return Q1 - (1.5*IQR), Q3 + (1.5*IQR) # Lower and upper bounds
    

    df_car_raw.drop(df_car_raw[(df_car_raw['Total distance travelled (lifetime) (km)']==0.0) | (df_car_raw['Total fuel consumed (lifetime) (l)']==0.0)].index.values, axis=0, inplace = True)
    df_van_raw.drop(df_van_raw[(df_van_raw['Total distance travelled (lifetime) (km)']==0.0) | (df_van_raw['Total fuel consumed (lifetime) (l)']==0.0)].index.values, axis=0, inplace = True)
    car_l_bound, car_u_bound = trim_data(df_car_raw['Total distance travelled (lifetime) (km)'])
    df_car_raw_trim = df_car_raw[(df_car_raw['Total distance travelled (lifetime) (km)']>car_l_bound) & (df_car_raw['Total distance travelled (lifetime) (km)']<car_u_bound)] .copy()
    van_l_bound, van_u_bound = trim_data(df_van_raw['Total distance travelled (lifetime) (km)'])
    df_van_raw_trim = df_van_raw[(df_van_raw['Total distance travelled (lifetime) (km)']>van_l_bound) & (df_van_raw['Total distance travelled (lifetime) (km)']<van_u_bound)].copy()

    car_fuel_distance_obj = FuelDistanceLR()
    car_residuals = car_fuel_distance_obj(df_car_raw_trim)
    van_fuel_distance_obj = FuelDistanceLR()
    van_residuals = van_fuel_distance_obj(df_van_raw_trim)
    coeffs_dict = {}
    intercept , coefficient = car_fuel_distance_obj.get_weights()
    coeffs_dict['car'] = {'intercept': intercept[0], 'coefficients': coefficient[0][0]} 
    intercept , coefficient = van_fuel_distance_obj.get_weights()
    coeffs_dict['van'] = {'intercept': intercept[0], 'coefficients': coefficient[0][0]} 
    print(car_fuel_distance_obj.get_metrics())
    print(van_fuel_distance_obj.get_metrics())
    # Create line data points
    line_car_data = []
    line_van_data = []
    for point_x in range(1,300000, 100):
        line_car_data.append(create_line_data_points(point_x,coeffs_dict['car']['intercept'], coeffs_dict['car']['coefficients']))
        line_van_data.append(create_line_data_points(point_x,coeffs_dict['van']['intercept'], coeffs_dict['van']['coefficients']))

    df_car_raw_trim['residuals'] = car_residuals
    df_van_raw_trim['residuals'] = van_residuals
    df_car_raw_trim['type'] = 'car'
    df_van_raw_trim['type'] = 'van'
    df_global = pd.concat([df_car_raw_trim, df_van_raw_trim], axis=0)
    df_global = df_global.rename(
        columns={
            'Total fuel consumed (lifetime) (l)': 'fuel_consumed',
            'Total distance travelled (lifetime) (km)': 'distance_travelled',
        }
    )
    df_cleaned = df_global.dropna(subset=['fuel_consumed', 'distance_travelled'])
    df_no_duplicates = df_cleaned.drop_duplicates(subset=['fuel_consumed', 'distance_travelled'])
    
    fig = go.Figure()
    data_cars = df_no_duplicates[df_no_duplicates['type']=='car'].copy()
    data_vans = df_no_duplicates[df_no_duplicates['type']=='van'].copy()
# Add traces. Scattergl for high volume of data
    fig.add_trace(go.Scattergl(x=data_cars['distance_travelled'].values, y=data_cars['fuel_consumed'].values,opacity =0.6,
        mode='markers',
        name='Cars',
        marker = dict(color='rgb(56,41,131)', size=3) # trendline='ols'
        ))
    fig.add_trace(go.Scattergl(x=data_vans['distance_travelled'].values, y=data_vans['fuel_consumed'].values,opacity =0.6,
        mode='markers',
        name='Vans',
        marker = dict(color='rgb(196,166,44)',size=3)
        ))
    fig.add_trace(go.Scatter(x=np.arange(1,300000,100), y=line_car_data, line = dict(color='rgb(67,178,47)', width=1.5, dash='dashdot'),
        mode='lines',
        name='Car-trend')
        )
    fig.add_trace(go.Scatter(x=np.arange(1,300000,100), y=line_van_data, line = dict(color='rgb(178,67,47)', width=1.5, dash='dashdot'),
        mode='lines',
        name='Van-trend')
    )
    fig.add_annotation(
        x=50000, # 175k
        y=create_line_data_points(50000,coeffs_dict['car']['intercept'], coeffs_dict['car']['coefficients']),
        text=f'Car Slope: {round(coeffs_dict["car"]["coefficients"],4)}km/l',
        showarrow=True,
        xanchor='left',
        xshift=10,
)
    fig.add_annotation(
        x=50000,
        y=create_line_data_points(50000,coeffs_dict['van']['intercept'], coeffs_dict['van']['coefficients']),
        text=f'Van Slope: {round(coeffs_dict["van"]["coefficients"],4)}km/l',
        showarrow=True,
        xanchor='left',
        yanchor='bottom',
)
    fig.update_layout(
        xaxis_range=[0, 70000],  # All data is contained withing 0 and 200k
        yaxis_range=[0, 15000],
        xaxis_title='Distance Travelled (km)',
        yaxis_title='Fuel Consumption (l)',
        legend = dict(font = dict(family = 'Droid Sans', size = 10, color = 'black'),itemsizing='constant'),
        title={
            'text': 'Lifetime: Fuel Consumed vs Distance Travelled (Trimmed)',
            'y': 0.92,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
        },
    )
    fig.show()
    fig.write_image('results/scatter_trimmed.png')
    return data_cars, data_vans

def plot_residuals(df,vehicle_type):
    if vehicle_type == 'van':
        color = 'rgb(196,166,44)'
        
    else:
        color = 'rgb(56,41,131)'
    df.sort_values(by=['distance_travelled'], inplace= True)
    debug = 'point'
    fig = go.Figure([
    go.Scattergl(
        x=df['distance_travelled'].values,
        y=df['residuals'].values,
        line=dict(color='rgb(0,100,80)'),
        mode='markers',
        opacity=.30,
        marker = dict(color=color,size=4)
    ),
    ])
    fig.update_layout(
        # xaxis_range=[0, 30000],  # All data is contained withing 0 and 200k
        xaxis_title='Distance Travelled (km)',
        yaxis_title='Residuals (l)',
        title={
            'text': f'Residuals for {vehicle_type}',
            'y': 0.92,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
        },
)   
    fig.show()
    fig.write_image(f'results/residuals_{vehicle_type}.png')