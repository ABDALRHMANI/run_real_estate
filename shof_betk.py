import os
import re
import base64
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import dash
from dash import dcc, html, Input, Output, State


# the data 
combined_df=pd.read_csv("Assests\\real_estate_csv\\combined_df.csv")
combined_df_r=pd.read_csv("Assests\\real_estate_csv\\combined_df_r.csv")
copy2_df=pd.read_csv("Assests\\real_estate_csv\\copy2_df.csv")
map_df=pd.read_csv("Assests\\real_estate_csv\\map_df.csv")
size_df=pd.read_csv("Assests\\real_estate_csv\\size_df.csv")
down_pay_df=pd.read_csv("Assests\\real_estate_csv\\real_down_pay.csv")



# the functions we use with the main page 
property_types_drop = ["Apartment", "Chalet", "Duplex", "Penthouse", "Townhouse", "Twin House", "Villa", "iVilla","All"]
drop_list_regions=combined_df['Region'].value_counts()
drop_box_values=drop_list_regions[~(drop_list_regions <= 10)].index
print(len(drop_box_values))

def map_figure_cus(df,region=None,prop_t=None):
    
    if region != None:
        if prop_t != 'All':
            df=df[(df['Region'] == region) & (df['Property Type'] == prop_t)]
        else:
            df=df[(df['Region'] == region)]
    fig = px.scatter_mapbox(
        df,
        lat="longitude",
        lon="latitude",
        hover_name="compound",
        color="price/met",
        hover_data=["price", "size_m", "price/met"],
        zoom=5,
        height=500,
        color_continuous_scale=px.colors.sequential.Inferno,
        range_color=[0, 150000]
    )

    fig.update_layout(mapbox_style="open-street-map")
    return fig
def Scatter_figure_cus(df,region=None,prop_t=None):
    if region != None:
        if prop_t != 'All':
            df=df[(df['Region'] == region) & (df['Property Type'] == prop_t)]
        else:
            df=df[(df['Region'] == region)]
    print(len(df))

    fig = px.scatter(df, x="size_m", y="price", facet_col="size_m_f", color="size_m_f")
    fig.update_xaxes(title="Size (m2)", range=[0, 500])
    fig.update_yaxes(title="Price", range=[0, 40000000])

    ann_len = len(df['size_m_f'].unique())
    names = ['0-50', '51-100', '101-150', '151-200', '201-250', '251-300', '301-350', '351-400', '401-450', '451-500']
    for i in range(ann_len):
        fig.layout.annotations[i]['text'] = names[i]

    # Move the legend to a different position
    fig.update_layout(
        legend=dict(
            x=1,  # Adjust x position to move legend to the right
            y=1,  # Adjust y position if needed
            traceorder='normal',
            font=dict(
                size=12,
            ),
            bgcolor='rgba(0,0,0,0)',  # Transparent background
        ),
        margin=dict(r=200)  # Increase right margin to accommodate legend
    )
    
    return fig
def bar_figure_cus(df,region=None,prop_t=None):
    if region != None:
        if prop_t != 'All':
            df=df[(df['Region'] == region) & (df['Property Type'] == prop_t)]
        else:
            df=df[(df['Region'] == region)]
    
    prop_count = len(df)
    property_types = ["Apartment", "Chalet", "Duplex", "Penthouse", "Townhouse", "Twin House", "Villa", "iVilla"]

    top_property_types = df['Property Type'].value_counts().head(5).index

    filtered_df = df[df['Property Type'].isin(top_property_types)]
    print(filtered_df.info())

    # Create the pivot table
    pivot_df = pd.pivot_table(filtered_df, values='price', index=['Bedrooms', 'Property Type'], aggfunc='median')
    pivot_df = pivot_df.reset_index()

    fig = px.bar(pivot_df, x='Bedrooms', y='price', facet_col='Property Type', width=1500, color='Property Type')

    fig.update_layout(barmode='group',
                      title='Median Price by Bedrooms and Property Type',
                      xaxis_title='Bedrooms', yaxis_title='Median Price',
                      template='plotly_white',
                      bargap=0.001,
                      )

    prop_len = len(df['Property Type'].unique())
    ann_list=pivot_df['Property Type'].unique()
    annotations = list(fig.layout.annotations)
    
    if prop_len == 1:
        annotations[0]['text'] = prop_t
    else:
        for i in range(5):
            if i < len(annotations):
                annotations[i]['text'] = ann_list[i]

    fig.layout.annotations = annotations
    
    return fig
def bar_down_pay_fig_cus(df,region=None,prop_t=None):
    if region != None:
        if prop_t != 'All':
            df=df[(df['Region'] == region) & (df['Property Type'] == prop_t)]
        else:
            df=df[(df['Region'] == region)]
    pivot = pd.pivot_table(df, index=['Down payment_f', 'installment_years_f'], aggfunc='size')
    pivot = pivot.rename("count").reset_index()
    
    category_mapping = {
    "(0, 500000]": "0-0.5M",
    "(500000, 1000000]": "0.5-1M",
    "(1000000, 1500000]": "1-1.5M",
    "(1500000, 2000000]": "1.5-2M",
    "(2000000, 2500000]": "2-2.5M",
    "(2500000, 3000000]": "2.5-3M",
    "(3000000, 3500000]": "3-3.5M",
    "(3500000, 4000000]": "3.5-4M",
    "(4000000, 4500000]": "4-4.5M",
    "(4500000, 5000000]": "4.5-5M",
    "(5000000, 5500000]": "5-5.5M",
    "(5500000, 6000000]": "5.5-6M",
    "(6000000, 6500000]": "6-6.5M"
    }

    pivot['Down payment_f'] = pivot['Down payment_f'].astype(str)
    pivot['Down payment_f'] = pivot['Down payment_f'].replace(category_mapping)

    color_mapping = {
        "1 year": "#1f77b4",
        "2 years": "#ff7f0e",
        "3 years": "#2ca02c",
        "4 years": "#d62728",
        "5 years": "#9467bd",
        "6 years": "#8c564b",
        "7 years": "#e377c2",
        "8 years": "#7f7f7f",
        "9 years": "#bcbd22",
        "10 years": "#17becf"
    }
    # Sort the categories in ascending order  
    sorted_categories = sorted(category_mapping.values())

    fig = px.bar(pivot, x="Down payment_f", y="count", color="installment_years_f", barmode='group',
                 color_discrete_map=color_mapping,
                 category_orders={"Down payment_f": sorted_categories})  # Add this line
    fig.update_layout(xaxis_title="Down Payment (Millions)")  # Update x-axis title
    
    return fig 


def point_regions_cus(df,region=None,prop_t=None):
  if region != None:
    if prop_t != 'All':
            df=df[(df['Region'] == region) & (df['Property Type'] == prop_t)]
    else:
        df=df[(df['Region'] == region)]

  pivot_comp=pd.pivot_table(df, values='price/met', index=['spec_comp'], aggfunc='median')

  pivot_comp['house_count']=df.groupby(['spec_comp']).size()


  filtered_pivot_comp = pivot_comp[pivot_comp['house_count'] >= 10]
  number_of_values=int(len(filtered_pivot_comp) / 2)
  if number_of_values > 15:
      number_of_values=15
      

  

    # Get largest and smallest regions by median price/met
  largest_regions = filtered_pivot_comp.nlargest(number_of_values, columns='price/met').sort_values(by='price/met', ascending=True)
  smallest_regions = filtered_pivot_comp.nsmallest(number_of_values, columns='price/met')

    # Create subplots
  fig = make_subplots(rows=2, cols=1,vertical_spacing=0.5, subplot_titles=(f"Top {number_of_values} compounds by Price/Met", f"Bottom {number_of_values} compounds by Price/Met"))

    # Plot largest regions
  fig.add_trace(
        go.Scatter(x=largest_regions.index, y=largest_regions['price/met'], mode='markers+lines', name='Largest compounds'),
        row=1, col=1
  )

    # Plot smallest regions
  fig.add_trace(
        go.Scatter(x=smallest_regions.index, y=smallest_regions['price/met'], mode='markers+lines', name='Smallest compounds'),
        row=2, col=1
  )

    # Update layout
  fig.update_layout(
        title_text="compounds Price/Met Analysis",
        height=500,
  )

  fig.update_xaxes(row=1, col=1)
  fig.update_yaxes(title_text="Price/Met", row=1, col=1)

  fig.update_xaxes(title_text="compounds", row=2, col=1)
  fig.update_yaxes(title_text="Price/Met", row=2, col=1)


  return fig
def point_regions(df):

  pivot_comp=pd.pivot_table(df, values='price/met', index=['Region'], aggfunc='median')

  pivot_comp['house_count']=df.groupby(['Region']).size()


  filtered_pivot_comp = pivot_comp[pivot_comp['house_count'] >= 10]

  largest_regions = filtered_pivot_comp.nlargest(10, columns='price/met').sort_values(by='price/met', ascending=True)
  smallest_regions = filtered_pivot_comp.nsmallest(10, columns='price/met')

  # Create subplots
  fig = make_subplots(rows=2,vertical_spacing=0.5, cols=1, subplot_titles=("Top 10 Regions by Price/Met", "Bottom 10 Regions by Price/Met"))

  # Plot largest regions
  fig.add_trace(
      go.Scatter(x=largest_regions.index, y=largest_regions['price/met'], mode='markers+lines', name='Largest Regions'),
      row=1, col=1
  )

  # Plot smallest regions
  fig.add_trace(
      go.Scatter(x=smallest_regions.index, y=smallest_regions['price/met'], mode='markers+lines', name='Smallest Regions'),
      row=2, col=1
  )

  # Update layout
  fig.update_layout(
      title_text="Region Price/Met Analysis",
      height=500,
      width=1300
  )

  fig.update_xaxes(row=1, col=1)
  fig.update_yaxes(title_text="Price/Met", row=1, col=1)

  fig.update_xaxes(title_text="Region", row=2, col=1)
  fig.update_yaxes(title_text="Price/Met", row=2, col=1)

  return fig

def median_price(df,region,prop_t):
    if prop_t != 'All':
        df=df[(df['Region'] == region) & (df['Property Type'] == prop_t)]
    else:
      df=df[(df['Region'] == region)]
    prop_count=len(df)
    return df['price/met'].median(),prop_count

    
def median_months(comb_df,comb_df_r,region,prop_t):
    try:
      if prop_t != 'All':
          comb_df=comb_df[(comb_df['Region'] == region) & (comb_df['Property Type'] == prop_t)]
          comb_df_r=comb_df_r[(comb_df_r['Region'] == region) & (comb_df_r['Property Type'] == prop_t)]
      else:
          comb_df=comb_df[(comb_df['Region'] == region)]
          comb_df_r=comb_df_r[(comb_df_r['Region'] == region)]
    except:
        return 'no enough data'
    price_of_buy_m=comb_df['price'].median()
    price_of_rent_m=comb_df_r['price'].median()
    return round((price_of_buy_m / price_of_rent_m) / 12,2)


drop_list_regions=combined_df_r['Region'].value_counts()
drop_box_values_rent=drop_list_regions[~(drop_list_regions <= 10)].index
def scatter_fig_rent(df,region=None,prop_t=None):
    if region != None:
        if prop_t != 'All':
            df=df[(df['Region'] == region) & (df['Property Type'] == prop_t)]
        else:
            df=df[(df['Region'] == region)]
    fig = px.scatter(df, x="size_m", y="price",facet_col="size_m_f",color="size_m_f")
    fig.update_xaxes(title="Size (m2)",range=[0,2000])
    fig.update_yaxes(title="Rent",range=[0,1000000])
    fig.update_layout(title="Rent vs Size (m2)")
    ran=len(df['size_m_f'].unique())
    names_r = ['0-200', '201-400', '401-600', '601-800', '801-1000', '1001-1200', '1201-1400', '1401-1600', '1601-1800', '1801-2000']
    for i in range(ran):
        fig.layout.annotations[i]['text'] = names_r[i]
    return fig

def bar_plot_rent(df, region=None, prop_t=None):
    if region is not None:
        if prop_t != 'All':
            df = df[(df['Region'] == region) & (df['Property Type'] == prop_t)]
        else:
            df = df[(df['Region'] == region)]

    property_types = ["Apartment", "Chalet", "Duplex", "Penthouse", "Townhouse", "Twin House", "Villa", "iVilla"]

    top_property_types = df['Property Type'].value_counts().head(5).index

    filtered_df = df[df['Property Type'].isin(top_property_types)]
    # Create the pivot table
    pivot_df = pd.pivot_table(filtered_df, values='price', index=['Bedrooms', 'Property Type'], aggfunc='median')

    pivot_df = pivot_df.reset_index()

    fig = px.bar(pivot_df, x='Bedrooms', y='price', facet_col='Property Type', color='Property Type')

    fig.update_layout(barmode='group',
                      title='Median Rent by Bedrooms and Property Type',
                      xaxis_title='Bedrooms', yaxis_title='Median Rent',
                      template='plotly_white',
                      height=500,
                      bargap=0.001,
                      )

    prop_len = len(df['Property Type'].unique())
    ann_list=pivot_df['Property Type'].unique()
    
    annotations = list(fig.layout.annotations)
    
    if prop_len == 1:
        annotations[0]['text'] = prop_t
    else:
        for i in range(prop_len):
            if i < len(annotations):
                annotations[i]['text'] = ann_list[i]

    fig.layout.annotations = annotations
    
    return fig
def region_plot_rent(df,region=None,prop_t=None):
    if region != None:
        if prop_t != 'All':
            df=df[(df['Region'] == region) & (df['Property Type'] == prop_t)]
        else:
            df=df[(df['Region'] == region)]
    
    pivot_comp=pd.pivot_table(df, values='price', index=['Region'], aggfunc='median')

    pivot_comp['house_count']=df.groupby(['Region']).size()


    filtered_pivot_comp = pivot_comp[pivot_comp['house_count'] >= 10]

    largest_regions = filtered_pivot_comp.nlargest(10, columns='price').sort_values(by='price', ascending=True)
    smallest_regions = filtered_pivot_comp.nsmallest(10, columns='price')

    # Create subplots
    fig = make_subplots(rows=2,vertical_spacing=0.5, cols=1, subplot_titles=("Top 10 Regions by Rent", "Bottom 10 Regions by Rent"))

    # Plot largest regions
    fig.add_trace(
        go.Scatter(x=largest_regions.index, y=largest_regions['price'], mode='markers+lines', name='Largest Regions'),
        row=1, col=1
    )

    # Plot smallest regions
    fig.add_trace(
        go.Scatter(x=smallest_regions.index, y=smallest_regions['price'], mode='markers+lines', name='Smallest Regions'),
        row=2, col=1
    )

    # Update layout
    fig.update_layout(
        title_text="Region Price Analysis",
        height=500,
        width=1300
    )

    fig.update_xaxes(row=1, col=1)
    fig.update_yaxes(title_text="rent", row=1, col=1)

    fig.update_xaxes(title_text="Region", row=2, col=1)
    fig.update_yaxes(title_text="rent", row=2, col=1)
    
    return fig
def point_regions_cus_rent(df,region=None,prop_t=None):
  if region != None:
    if prop_t != 'All':
            df=df[(df['Region'] == region) & (df['Property Type'] == prop_t)]
    else:
        df=df[(df['Region'] == region)]

  pivot_comp=pd.pivot_table(df, values='price', index=['spec_Compound'], aggfunc='median')

  pivot_comp['house_count']=df.groupby(['spec_Compound']).size()


  filtered_pivot_comp = pivot_comp[pivot_comp['house_count'] >= 10]
  number_of_values=int(len(filtered_pivot_comp) / 2)
  if number_of_values > 15:
      number_of_values=15
      

  

    # Get largest and smallest regions by median price/met
  largest_regions = filtered_pivot_comp.nlargest(number_of_values, columns='price').sort_values(by='price', ascending=True)
  smallest_regions = filtered_pivot_comp.nsmallest(number_of_values, columns='price')

    # Create subplots
  fig = make_subplots(rows=2, cols=1,vertical_spacing=0.5, subplot_titles=(f"Top {number_of_values} compounds by Rent", f"Bottom {number_of_values} compounds by Rent"))

    # Plot largest regions
  fig.add_trace(
        go.Scatter(x=largest_regions.index, y=largest_regions['price'], mode='markers+lines', name='Largest compounds'),
        row=1, col=1
  )

    # Plot smallest regions
  fig.add_trace(
        go.Scatter(x=smallest_regions.index, y=smallest_regions['price'], mode='markers+lines', name='Smallest compounds'),
        row=2, col=1
  )

    # Update layout
  fig.update_layout(
        title_text="compounds Rent Analysis",
        height=500,
  )

  fig.update_xaxes(row=1, col=1)
  fig.update_yaxes(title_text="Rent", row=1, col=1)

  fig.update_xaxes(title_text="compounds", row=2, col=1)
  fig.update_yaxes(title_text="Rent", row=2, col=1)


  return fig
def median_rent(df,region,prop_t):
    if prop_t != 'All':
        df=df[(df['Region'] == region) & (df['Property Type'] == prop_t)]
    else:
      df=df[(df['Region'] == region)]
    prop_count=len(df)
    return df['price'].median(),prop_count




# the code of the dash board
import dash
from dash import dcc, html, Input, Output,State
import base64
import plotly.express as px

# Initialize the Dash app
app = dash.Dash(__name__,suppress_callback_exceptions=True)
server=app.server
# Convert image to base64 string
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()

encoded_image = encode_image("E:\\icons\\mylogo3.png")
encoded_image_map_icon = encode_image("E:\\icons\\map_icon.jpeg")
encoded_image_pricing_icon = encode_image("E:\\icons\\pricing-icon.jpg")
encoded_image_size_icon = encode_image("E:\\icons\\size_icon.png")
encoded_image_scatter_icon = encode_image("E:\\icons\\scatter-icon.jpg")
encoded_image_bar_icon = encode_image("E:\\icons\\bar_graph_icon.png")
encoded_image_bedrooms_icon = encode_image("E:\\icons\\bedrooms_icon.png")
encoded_image_property_type_icon = encode_image("E:\\icons\\property_type_icon.jpeg")
encoded_image_group_bar_icon = encode_image("E:\\icons\\group_bar_icon.png")
encoded_image_sell_con = encode_image("E:\\icons\\sell_icon.png")
encoded_image_rent_con = encode_image("E:\\icons\\rent_house.png")
encoded_image_region_con = encode_image("E:\\icons\\regions_icon.jpeg")
# Define the layout of the main page
app.title = "شوف بيتك"  # Set the title in Arabic
app.layout = html.Div([
    html.Nav(
        className="bg-black p-4 rounded-lg border border-gray-300",
        children=[
            html.Div(
                className="container mx-auto flex items-center justify-between",
                children=[
                    html.Div(
                        html.Img(
                            src=f"data:image/png;base64,{encoded_image}",
                            className="h-20 rounded-lg border border-gray-300 mr-4"
                        ),
                    ),
                    html.Div(
                        className="flex space-x-4",
                        children=[
                            html.Div(
                                className="relative group",
                                children=[
                                    html.A(
                                        ["General", html.Span(" ▼", className="ml-1")],
                                        href="#",
                                        className="text-white hover:text-blue-300",
                                        id="link-general_1"
                                    ),
                                    html.Div(
                                        className="dropdown-content absolute hidden bg-black text-white mt-2 py-2 rounded-lg border border-gray-300",
                                        children=[
                                            html.A("General Sell Data", href="#", className="block px-4 py-2 hover:bg-gray-700", id="link-general-sell"),
                                            html.A("General Rent Data", href="#", className="block px-4 py-2 hover:bg-gray-700", id="link-general-rent"),
                                        ],
                                    ),
                                ],
                            ),
                            html.Div(
                                className="relative group",
                                children=[
                                    html.A(
                                        ["Customize\comparison", html.Span(" ▼", className="ml-1")],
                                        href="#",
                                        className="text-white hover:text-blue-300",
                                        id="link-general"
                                    ),
                                    html.Div(
                                        className="dropdown-content absolute hidden bg-black text-white mt-2 py-2 rounded-lg border border-gray-300",
                                        children=[
                                            html.A("Customize\comparison Sell Data", href="#", className="block px-4 py-2 hover:bg-gray-700", id="link-Customize-sell"),
                                            html.A("Customize\comparison Rent Data", href="#", className="block px-4 py-2 hover:bg-gray-700", id="link-Customize-rent"),
                                        ],
                                    ),
                                ],
                            ),
                        ],
                    ),
                ],
            ),
        ],
    ),
    html.Div(id="page-content"),
], className="px-4 py-8")

def main_page_layout():
    return html.Div([
        html.Div([
            html.Div([
                html.Img(src=f"data:image/jpeg;base64,{encoded_image_sell_con}", className="stat-item-img", style={"height": "50px", "width": "50px"}),
                html.H3("44000", className="number"),
                html.P("Number of properties for sale", className="label"),
            ], className="stat-item"),
            html.Div([
                html.Img(src=f"data:image/jpeg;base64,{encoded_image_rent_con}", className="stat-item-img", style={"height": "50px", "width": "50px"}),
                html.H3("28000", className="number"),
                html.P("Number of properties for rent", className="label"),
            ], className="stat-item"),
            html.Div([
                html.Img(src=f"data:image/jpeg;base64,{encoded_image_region_con}", className="stat-item-img", style={"height": "50px", "width": "50px"}),
                html.H3("45", className="number"),
                html.P("Regions", className="label"),
            ], className="stat-item"),
            html.Div([
                html.Img(src=f"data:image/jpeg;base64,{encoded_image_property_type_icon}", className="stat-item-img", style={"height": "50px", "width": "50px"}),
                html.H3("8", className="number"),
                html.P("Property Types", className="label"),
            ], className="stat-item"),
        ], className="stats-section"),
        html.Div([
            html.H1("Explore Real Estate in Egypt", className="text-4xl font-bold mt-8 text-gray-800"),
            html.P("The better place to explore real estate in Egypt", className="text-lg mt-4 text-gray-600"),
        ], className="jumbotron text-center"),

        # First Section
        html.Div(className="section-container", children=[
            html.Br(),
            html.Div([
                html.Div([
                    html.H1([
                        html.Img(src=f"data:image/jpeg;base64,{encoded_image_map_icon}", className="h-12", style={"marginRight": "10px"}),
                        " map_figure:", 
                    ], className="text-4xl font-semibold mt-6 flex items-center", style={"font-size": "2rem"}),
                    html.H2([
                        "The main goal is to visually analyze how ",
                        html.Span("location", style={"color": "#3896FF"}),
                        " influences the ",
                        html.Span("price per meter", style={"color": "olive"}),
                        "."
                    ], className="text-2xl font-semibold mt-8 text-gray-800", style={"padding": "10px 0"}),
                    html.P([
                        "To visualize changes in price per meter, locations with ",
                        html.Span("darker colors", style={
                            "display": "inline-block",
                            "width": "20px",
                            "height": "20px",
                            "backgroundColor": px.colors.sequential.Inferno[0],
                            "margin": "0 5px",
                            "borderRadius": "50%",
                            "verticalAlign": "middle",
                            "border": "1px solid #ccc",
                        }),
                        " indicate lower prices, while locations with ",
                        html.Span("lighter colors", style={
                            "display": "inline-block",
                            "width": "20px",
                            "height": "20px",
                            "backgroundColor": px.colors.sequential.Inferno[-1],
                            "margin": "0 5px",
                            "borderRadius": "50%",
                            "verticalAlign": "middle",
                            "border": "1px solid #ccc",
                        }),
                        " indicate higher prices."
                    ], className="text-lg font-semibold mt-5 text-gray-800", style={"padding": "10px 0"}),
                    html.H3("How to Use the Map:", className="text-xl font-semibold mt-8 text-gray-800", style={"padding": "10px 0"}),
                    html.P([
                        "1. ", html.Span("Zoom In and Out: ", className="font-semibold"), "Use the zoom controls to focus on specific regions. This allows you to see more detailed information about the area you are interested in.",
                        html.Br(),
                        "2. ", html.Span("Navigate: ", className="font-semibold"), "Click and drag with your mouse to move around the map. This way, you can explore different regions and observe the changes in price per meter.",
                        html.Br(),
                        "3. ", html.Span("Explore Data Points: ", className="font-semibold"), "Hover over any point on the map to see detailed information about that location, including the price per meter. This feature helps you discover specific data for each point on the map."
                    ], className="text-lg mt-4 text-gray-600", style={"padding": "10px 0"})
                ], className="map_description", style={"padding": "20px"}),
                dcc.Graph(
                    id='map-figure',
                    figure=map_figure_cus(map_df),  # Define this function or replace with your map figure function
                    className="mt-8"
                ),
            ]),
        ]),

        # Second Section
        html.Div(className="section-container", children=[
            html.Div([
                html.H1([
                    html.Img(src=f"data:image/jpeg;base64,{encoded_image_scatter_icon}", style={"height": "80px", "width": "100px", "marginRight": "10px"}),
                    "scatter plot between size and price"
                ], className="text-4xl font-semibold mt-6 flex items-center", style={"font-size": "2rem"}),
                html.H3([
                    "The main goal is to understand the relationship between",
                    html.Br(),
                    html.Span([
                        "the size ",
                        html.Img(
                            src=f"data:image/jpeg;base64,{encoded_image_size_icon}",
                            style={"height": "25px", "width": "25px", "marginRight": "10px", "verticalAlign": "middle", "display": "inline-block"}
                        ),
                    ], className="size-text", style={"display": "inline-block", "verticalAlign": "middle"}),                
                    html.Br(),
                    "and",
                    html.Br(),
                    html.Span([
                        "the price of the house",
                        html.Img(
                            src=f"data:image/jpeg;base64,{encoded_image_pricing_icon}",
                            style={"height": "25px", "width": "25px", "marginRight": "10px", "verticalAlign": "middle", "display": "inline-block"}
                        ),
                    ], className="price-text", style={"display": "inline-block", "verticalAlign": "middle"})
                ], className="text-xl font-semibold mt-8 text-gray-800"),
                html.H4("How to Use It:", className="text-lg font-semibold mt-4 text-gray-800"),
                html.P([
                    "1. ", html.Span("Select a Size Category: ", className="font-semibold"), "Choose a specific size category to explore its impact on price.",
                    html.Br(),
                    "2. ", html.Span("Specify a Price Range: ", className="font-semibold"), "Determine a specific price range to investigate."
                ], className="text-lg mt-4 text-gray-600")
            ], className="size_description"),
            dcc.Graph(
                id='scatter-figure',
                figure=Scatter_figure_cus(size_df),
                className="mt-8"
            ),
        ]),

        # Third Section
        html.Div(className="section-container", children=[
            html.Div([
                html.H1([
                    html.Img(src=f"data:image/jpeg;base64,{encoded_image_bar_icon}", style={"height": "50px", "width": "50px", "marginRight": "10px"}),
                    "bar plot between (bedrooms categorized by property type) with the price"
                ], className="text-2xl font-semibold mt-6 flex items-center", style={"font-size": "2rem"}),
                html.H3([
                    "The main goal is to understand the relationship between:",
                    html.Br(),
                    html.Span([
                        "property type ",
                        html.Img(
                            src=f"data:image/jpeg;base64,{encoded_image_property_type_icon}",
                            style={"height": "50px", "width": "50px", "marginRight": "10px", "verticalAlign": "middle", "display": "inline-block"}
                        ),
                    ], className="size-text", style={"display": "inline-block", "verticalAlign": "middle"}),
                    html.Br(),
                    "and",
                    html.Br(),
                    html.Span([
                        "the number of bedrooms ",
                        html.Img(
                            src=f"data:image/jpeg;base64,{encoded_image_bedrooms_icon}",
                            style={"height": "50px", "width": "50px", "marginLeft": "10px", "verticalAlign": "middle", "display": "inline-block"}
                        ),
                    ], className="size-text", style={"display": "inline-block", "verticalAlign": "middle"}),
                    html.Br(),
                    "on",
                    html.Br(),
                    html.Span([
                        "the price of the house",
                        html.Img(
                            src=f"data:image/jpeg;base64,{encoded_image_pricing_icon}",
                            style={"height": "25px", "width": "25px", "marginLeft": "10px", "verticalAlign": "middle", "display": "inline-block"}
                        ),
                    ], className="price-text", style={"display": "inline-block", "verticalAlign": "middle"})
                ], className="text-xl font-semibold mt-8 text-gray-800"),
                html.H4("How to Use It:", className="text-lg font-semibold mt-4 text-gray-800"),
                html.P([
                    "1. ", html.Span("Select a Property Type Category: ", className="font-semibold"), "Choose a specific property type category to explore its impact on price.",
                    html.Br(),
                    "2. ", html.Span("Get the Median of the Bar: ", className="font-semibold"), "You can hover over any specific bar to see the exact median price for that number of bedrooms and property type."
                ], className="text-lg mt-4 text-gray-600")
            ], className="bedrooms_description"),
            dcc.Graph(
                id='bar-figure',
                figure=bar_figure_cus(copy2_df),
                className="mt-8"
            ),
        ]),

        # Fourth Section
        html.Div(className="section-container", children=[
            html.Div([
                html.H1([
                    html.Img(src=f"data:image/jpeg;base64,{encoded_image_group_bar_icon}", style={"height": "50px", "width": "50px", "marginRight": "10px"}),
                    "Analyzing Downpayment and Installment Year Distributions in Egypt's Real Estate Market with group bar"
                ], className="text-2xl font-semibold mt-6 flex items-center", style={"font-size": "1.5rem"}),
                html.H4("How to Use:", className="text-lg font-semibold mt-4 text-gray-800"),
                html.P([
                    "1. ", html.Span("Explore Distribution: ", className="font-semibold"), 
                    "Each group of bars shows house counts across different down payment ranges. Within each group, bars represent installment years and their corresponding house counts.",
                    html.Br(),
                    "2. ", html.Span("Hover for Details: ", className="font-semibold"), 
                    "Hover over any bar to see the exact number of houses for each installment year within the selected down payment range."
                ], className="text-lg mt-4 text-gray-600")
            ]),
            dcc.Graph(
                id='bar_down_pay_fig',
                figure=bar_down_pay_fig_cus(down_pay_df),
                className="mt-8"
            ),
        ]),

        # Fifth Section
        html.Div(className="section-container", children=[
            html.Div([
                html.H1([ 
                    html.Span("Top Regions", style={"color": "green"}), 
                    " and ", 
                    html.Span("Bottom Regions", style={"color": "red"}), 
                    " in Price per Meter in Egypt"
                ], className="text-3xl font-semibold mt-4 text-gray-800"),            
                html.H3("Observe the price per meter differences across regions in Egypt to assist your search.", className="text-xl font-semibold mt-8 text-gray-800"),
                html.H4("How to Use:", className="text-lg font-semibold mt-4 text-gray-800"),
                html.P([
                    "1. ", html.Span("Observe the Distribution: ", className="font-semibold"), "The height of each point represents the price per meter for each region. Notice the differences to gain insights.",
                    html.Br(),
                    "2. ", html.Span("Hover for Details: ", className="font-semibold"), "Hover over any point to see the region name and price per meter. Zoom in to focus on specific areas."
                ], className="text-lg mt-4 text-gray-600")
            ]),
            dcc.Graph(
                id='point_regions',
                figure=point_regions(combined_df),
                className="mt-8"
            ),
        ]),
    ])
    
    

def cus_rent():
    return html.Div([
        html.H1("Customize/Comparison Rent Page", className="text-4xl font-bold mt-8 text-gray-800"),
        html.P("Here you can customize your comparison settings.", className="text-lg mt-4 text-gray-600"),
        html.Br(),
        # Horizontal layout for dropdowns and vertical line
        html.Div([
            html.Div([
                html.Label("Select Region:", className="block text-gray-700 text-sm font-bold mb-2"),
                dcc.Dropdown(
                    id='region-dropdown-left_rent',
                    options=[{'label': region, 'value': region} for region in drop_box_values_rent],
                    placeholder="Select regions...",
                    style={"width": "100%"}  # Adjust width of the dropdown
                ),
                html.Label("Select Property Type:", className="block text-gray-700 text-sm font-bold mb-2 mt-4"),
                dcc.Dropdown(
                    id='property-dropdown-left_rent',
                    options=[{'label': property, 'value': property} for property in property_types_drop],
                    placeholder="Select the type...",
                    style={"width": "100%"}  # Adjust width of the dropdown
                )
            ], style={"width": "45%", "display": "inline-block"}),  # Adjust width and display inline-block

            # Vertical line
            html.Div(style={"border-left": "5px solid #ccc", "height": "150px", "display": "inline-block", "vertical-align": "top", "margin-left": "20px", "border-left-color": "#01A0D3"}),

            html.Div([
                html.Label("Select Region:", className="block text-gray-700 text-sm font-bold mb-2"),
                dcc.Dropdown(
                    id='region-dropdown-right_rent',
                    options=[{'label': region, 'value': region} for region in drop_box_values_rent],
                    placeholder="Select regions...",
                    style={"width": "100%"}  # Adjust width of the dropdown
                ),
                html.Label("Select Property Type:", className="block text-gray-700 text-sm font-bold mb-2 mt-4"),
                dcc.Dropdown(
                    id='property-dropdown-right_rent',
                    options=[{'label': property, 'value': property} for property in property_types_drop],
                    placeholder="Select the type...",
                    style={"width": "100%"}  # Adjust width of the dropdown
                )
            ], style={"width": "45%", "display": "inline-block", "margin-left": "20px"})  # Adjust width, display inline-block, and margin-left
        ], style={"text-align": "center", "margin-bottom": "20px"}),  # Center-align and add margin-bottom

        # Compare button above the line
        html.Div([
            html.Button("Compare between the two sides", id='compare-button_rent', className="bg-green-500 hover:bg-green-700 text-white font-bold py-2 px-4 rounded")
        ], className="flex justify-center mb-4"),

        # Filter button
        html.Button("Filter with the left side", id='custom-button_rent', className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded mb-4"),

        # Results sections
        html.Div(id='filtered-results_rent', className="mt-4"),
        html.Div(id='comparison-results_rent', className="mt-4", style={"width": "100%", "margin": "0 auto"})
    ], className="px-20 py-8")  # Use padding similar to customize_comparison_layou

def generate_result_components(region, property_type, dataframes):
    try:
        print(region)
        copy2_df, map_df, size_df, down_pay_df, combined_df, combined_df_r = dataframes
        median_price_n, prop_count = median_price(combined_df, region, property_type)
        print("fucking", prop_count)
        bar_fig = bar_figure_cus(copy2_df, region, property_type)
        map_fig = map_figure_cus(map_df, region, property_type)
        scat_fig = Scatter_figure_cus(size_df, region, property_type)
        down_pay_fig = bar_down_pay_fig_cus(down_pay_df, region, property_type)
        point_regions_fig = point_regions_cus(combined_df, region, property_type)
        
        number_of_months = median_months(combined_df, combined_df_r, region, property_type)

        colored_region = html.Span(region, style={"color": "#0290EC", "font-weight": "bold", "marginLeft": "5px", "marginRight": "5px"})
        colored_property = html.Span(property_type, style={"color": "#8E05F3", "font-weight": "bold", "marginLeft": "5px", "marginRight": "5px"})
        
        result_div = html.Div([
            html.P(f"Number of properties: {prop_count} House", className="mt-4 text-xl text-gray-900"),
            html.P(f"The median price/met: {median_price_n} EGP ", className="mt-4 text-xl text-gray-900"),
            html.P(f"The number of years required to recoup the cost of purchasing your house through rental income: {number_of_months} years ", className="mt-4 text-xl text-gray-900"),
            html.Div([
                html.H1([
                    html.Img(src=f"data:image/jpeg;base64,{encoded_image_map_icon}", className="h-12", style={"marginRight": "10px"}),
                    "Map Figure in", colored_region, "with", colored_property, "Property Type"
                ], className="text-xl font-semibold mt-6 flex items-center", style={"font-size": "1.5rem"}),
                dcc.Graph(figure=map_fig),
            ], className="section-container"),

            html.Div([
                html.H1([
                    html.Img(src=f"data:image/jpeg;base64,{encoded_image_scatter_icon}", style={"height": "80px", "width": "100px", "marginRight": "10px"}),
                    "Scatter Plot between Size and Price in", colored_region, "with", colored_property, "Property Type"
                ], className="text-l font-semibold mt-6 flex items-center", style={"font-size": "1.25rem"}),
                dcc.Graph(figure=scat_fig),
            ], className="section-container"),

            html.Div([
                html.H1([
                    html.Img(src=f"data:image/jpeg;base64,{encoded_image_bar_icon}", style={"height": "50px", "width": "50px", "marginRight": "10px"}),
                    "Bar Plot between Bedrooms Categorized by Property Type with the Price in", colored_region, "with", colored_property, "Property Type"
                ], className="text-l font-semibold mt-6 flex items-center", style={"font-size": "1rem"}),
                dcc.Graph(figure=bar_fig),
            ], className="section-container"),

            html.Div([
                html.H1([
                    html.Img(src=f"data:image/jpeg;base64,{encoded_image_group_bar_icon}", style={"height": "50px", "width": "50px", "marginRight": "10px"}),
                    "Analyzing Downpayment and Installment Year Distributions in", colored_region, "and", colored_property, "Property Type with Group Bar"
                ], className="text-l font-semibold mt-6 flex items-center", style={"font-size": "1rem"}),
                dcc.Graph(figure=down_pay_fig),
            ], className="section-container"),

            html.Div([
                html.H1([ 
                    html.Span("Top compounds", style={"color": "green"}), 
                    " and ", 
                    html.Span("Bottom compounds", style={"color": "red"}), 
                    "in Price per Meter in", colored_region, "and", colored_property, "property type"
                ], className="text-3xl font-semibold mt-4 text-gray-800"),
                dcc.Graph(figure=point_regions_fig),
            ], className="section-container"),
        ], style={"padding": "5px"})
        
        return result_div

    except Exception as e:
        print(f"Error occurred: {e}")
        return html.Div([
            html.P("There is no data available for the specified condition.", className="mt-4 text-xl text-gray-900")
        ], style={"padding": "5px"})


def generate_result_components_rent(region, property_type):
    print(region)
    # Generating figures and statistics
    scat_f_r = scatter_fig_rent(combined_df_r, region, property_type)
    bar_f_r = bar_plot_rent(combined_df_r, region, property_type)
    re_f_r = point_regions_cus_rent(combined_df_r, region, property_type)
    m_ren, prop_co = median_rent(combined_df_r, region, property_type)
    number_of_months = median_months(combined_df, combined_df_r, region, property_type)

    # Color styling for region and property type
    colored_region = html.Span(region, style={"color": "#0290EC", "font-weight": "bold", "marginLeft": "5px", "marginRight": "5px"})
    colored_property = html.Span(property_type, style={"color": "#8E05F3", "font-weight": "bold", "marginLeft": "5px", "marginRight": "5px"})

    result_div = html.Div([
        html.P(f"Number of properties: {prop_co} House", className="mt-4 text-xl text-gray-900"),
        html.P(f"The median Rent: {m_ren} EGP", className="mt-4 text-xl text-gray-900"),
        html.P(f"The number of years required to recoup the cost of purchasing your house through rental income: {number_of_months} years", className="mt-4 text-xl text-gray-900"),
        html.Div([
            html.H1([
                html.Img(src=f"data:image/jpeg;base64,{encoded_image_scatter_icon}", style={"height": "80px", "width": "100px", "marginRight": "10px"}),
                "Scatter Plot between Size and Price in", colored_region, "with", colored_property, "Property Type"
            ], className="text-l font-semibold mt-6 flex items-center", style={"font-size": "1.25rem"}),
            dcc.Graph(figure=scat_f_r),
        ], className="section-container"),

        html.Div([
            html.H1([
                html.Img(src=f"data:image/jpeg;base64,{encoded_image_bar_icon}", style={"height": "50px", "width": "50px", "marginRight": "10px"}),
                "Bar Plot between Bedrooms Categorized by Property Type with the Price in", colored_region, "with", colored_property, "Property Type"
            ], className="text-l font-semibold mt-6 flex items-center", style={"font-size": "1rem"}),
            dcc.Graph(figure=bar_f_r),
        ], className="section-container"),

        html.Div([
            html.H1([
                html.Span("Top compounds", style={"color": "green"}),
                " and ",
                html.Span("Bottom compounds", style={"color": "red"}),
                " in Price per Meter in", colored_region, "and", colored_property, "property type"
            ], className="text-3xl font-semibold mt-4 text-gray-800"),
            dcc.Graph(figure=re_f_r),
        ], className="section-container"),
    ], style={"padding": "5px"})
    
    return result_div

@app.callback(
    [Output('filtered-results_rent', 'children'),
     Output('comparison-results_rent', 'children')],
    [Input('custom-button_rent', 'n_clicks'),
     Input('compare-button_rent', 'n_clicks')],
    [State('region-dropdown-left_rent', 'value'),
     State('property-dropdown-left_rent', 'value'),
     State('region-dropdown-right_rent', 'value'),
     State('property-dropdown-right_rent', 'value')]
)
def update_results_rent(custom_clicks, compare_clicks, left_region, left_property, right_region, right_property):
    ctx = dash.callback_context
    if not ctx.triggered:
        return "", ""
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if button_id == 'custom-button_rent' and custom_clicks:
        if left_region and left_property:
            results = generate_result_components_rent(left_region, left_property)
            return results, ""
        else:
            return "Please select both region and property type for customization.", ""

    elif button_id == 'compare-button_rent' and compare_clicks:
        if left_region and left_property and right_region and right_property:
            left_results = generate_result_components_rent(left_region, left_property)
            right_results = generate_result_components_rent(right_region, right_property)

            comparison_div = html.Div([
            html.Div([
                html.H2("Left Side Comparison", className="text-3xl font-bold mb-4 text-gray-800"),
                html.Div(left_results, className="p-4 bg-white border border-gray-200 rounded shadow-md")
            ], className="w-1/2 pr-4", style={"flex": "0 0 85%"}),
            html.Div([
                html.H2("Right Side Comparison", className="text-3xl font-bold mb-4 text-gray-800"),
                html.Div(right_results, className="p-4 bg-white border border-gray-200 rounded shadow-md")
            ], className="w-1/2 pl-4", style={"flex": "0 0 85%"}),
        ], className="flex")

            return "", comparison_div
        else:
            return "", "Please select both region and property type for comparison."

    return "", ""

@app.callback(
    [Output('filtered-results', 'children'),
     Output('comparison-results', 'children')],
    [Input('custom-button', 'n_clicks'),
     Input('compare-button', 'n_clicks')],
    [Input('region-dropdown-left', 'value'),
     Input('property-dropdown-left', 'value'),
     Input('region-dropdown-right', 'value'),
     Input('property-dropdown-right', 'value')]
)
def update_results(custom_clicks, compare_clicks, left_region, left_property, right_region, right_property):
    ctx = dash.callback_context
    if not ctx.triggered:
        return "", ""
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    dataframes = (copy2_df, map_df, size_df, down_pay_df, combined_df, combined_df_r)
    
    if button_id == 'custom-button' and custom_clicks:
        if left_region and left_property:
            return generate_result_components(left_region, left_property, dataframes), ""
        else:
            return "Please select both region and property type for customization.", ""

    elif button_id == 'compare-button' and compare_clicks:
        if left_region and left_property and right_region and right_property:
            left_results = generate_result_components(left_region, left_property, dataframes)
            right_results = generate_result_components(right_region, right_property, dataframes)

            comparison_div = html.Div([
            html.Div([
                html.H2("Left Side Comparison", className="text-3xl font-bold mb-4 text-gray-800"),
                html.Div(left_results, className="p-4 bg-white border border-gray-200 rounded shadow-md")
            ], className="w-1/2 pr-4", style={"flex": "0 0 75%"}),
            html.Div([
                html.H2("Right Side Comparison", className="text-3xl font-bold mb-4 text-gray-800"),
                html.Div(right_results, className="p-4 bg-white border border-gray-200 rounded shadow-md")
            ], className="w-1/2 pl-4", style={"flex": "0 0 75%"}),
        ], className="flex")

            return "", comparison_div
        else:
            return "", "Please select both region and property type for comparison."

    return "", ""

def customize_comparison_layout():
    return html.Div([
        html.H1("Customize/Comparison Page", className="text-4xl font-bold mt-8 text-gray-800"),
        html.P("Here you can customize your comparison settings.", className="text-lg mt-4 text-gray-600"),
        html.Br(),
        
        # Horizontal layout for dropdowns and vertical line
        html.Div([
            html.Div([
                html.Label("Select Region:", className="block text-gray-700 text-sm font-bold mb-2"),
                dcc.Dropdown(
                    id='region-dropdown-left',
                    options=[{'label': region, 'value': region} for region in drop_box_values],
                    placeholder="Select regions...",
                    style={"width": "100%"}  # Adjust width of the dropdown
                ),
                html.Label("Select Property Type:", className="block text-gray-700 text-sm font-bold mb-2 mt-4"),
                dcc.Dropdown(
                    id='property-dropdown-left',
                    options=[{'label': property, 'value': property} for property in property_types_drop],
                    placeholder="Select the type...",
                    style={"width": "100%"}  # Adjust width of the dropdown
                )
            ], style={"width": "45%", "display": "inline-block"}),  # Adjust width and display inline-block

            # Vertical line
            html.Div(style={
                "border-left": "5px solid #ccc",
                "height": "150px",
                "display": "inline-block",
                "vertical-align": "top",
                "margin-left": "20px",
                "border-left-color": "#01A0D3"
            }),

            html.Div([
                html.Label("Select Region:", className="block text-gray-700 text-sm font-bold mb-2"),
                dcc.Dropdown(
                    id='region-dropdown-right',
                    options=[{'label': region, 'value': region} for region in drop_box_values],
                    placeholder="Select regions...",
                    style={"width": "100%"}  # Adjust width of the dropdown
                ),
                html.Label("Select Property Type:", className="block text-gray-700 text-sm font-bold mb-2 mt-4"),
                dcc.Dropdown(
                    id='property-dropdown-right',
                    options=[{'label': property, 'value': property} for property in property_types_drop],
                    placeholder="Select the type...",
                    style={"width": "100%"}  # Adjust width of the dropdown
                )
            ], style={"width": "45%", "display": "inline-block", "margin-left": "20px"})  # Adjust width, display inline-block, and margin-left
        ], style={"text-align": "center", "margin-bottom": "20px"}),  # Center-align and add margin-bottom

        # Compare button above the line
        html.Div([
            html.Button("Compare between the two sides", id='compare-button', className="bg-green-500 hover:bg-green-700 text-white font-bold py-2 px-4 rounded")
        ], className="flex justify-center mb-4"),

        # Filter button
        html.Button("Filter with the left side", id='custom-button', className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded mb-4"),

        # Placeholder for filtered results
        html.Div(id='filtered-results', className="mt-4"),
        html.Div(id='comparison-results', className="mt-4")
    ], className="px-4 py-8")
def rent_price_layout():
    return html.Div([
        
        html.Div([
            html.Div([
                html.Img(src=f"data:image/jpeg;base64,{encoded_image_sell_con}", className="stat-item-img", style={"height": "50px", "width": "50px"}),
                html.H3("44000", className="number"),
                html.P("Number of properties for sale", className="label"),
            ], className="stat-item"),
            html.Div([
                html.Img(src=f"data:image/jpeg;base64,{encoded_image_rent_con}", className="stat-item-img", style={"height": "50px", "width": "50px"}),
                html.H3("28000", className="number"),
                html.P("Number of properties for rent", className="label"),
            ], className="stat-item"),
            html.Div([
                html.Img(src=f"data:image/jpeg;base64,{encoded_image_region_con}", className="stat-item-img", style={"height": "50px", "width": "50px"}),
                html.H3("45", className="number"),
                html.P("Regions", className="label"),
            ], className="stat-item"),
            html.Div([
                html.Img(src=f"data:image/jpeg;base64,{encoded_image_property_type_icon}", className="stat-item-img", style={"height": "50px", "width": "50px"}),
                html.H3("8", className="number"),
                html.P("Property Types", className="label"),
            ], className="stat-item"),
        ], className="stats-section"),
        html.Div([
            html.H1("Explore Rent in egypt", className="text-4xl font-bold mt-8 text-gray-800"),
            html.P("The better place to explore Rent in Egypt", className="text-lg mt-4 text-gray-600"),
        ], className="jumbotron text-center"),
        html.Div(className="section-container", children=[
            html.Div([
                html.H1([
                    html.Img(src=f"data:image/jpeg;base64,{encoded_image_scatter_icon}", style={"height": "80px", "width": "100px", "marginRight": "10px"}),
                    "scatter plot between size and price"
                ], className="text-4xl font-semibold mt-6 flex items-center", style={"font-size": "2rem"}),
                html.H3([
                    "The main goal is to understand the relationship between",
                    html.Br(),
                    html.Span([
                        "the size ",
                        html.Img(
                            src=f"data:image/jpeg;base64,{encoded_image_size_icon}",
                            style={"height": "25px", "width": "25px", "marginRight": "10px", "verticalAlign": "middle", "display": "inline-block"}
                        ),
                    ], className="size-text", style={"display": "inline-block", "verticalAlign": "middle"}),                
                    html.Br(),
                    "and",
                    html.Br(),
                    html.Span([
                        "the Rent of the house",
                        html.Img(
                            src=f"data:image/jpeg;base64,{encoded_image_pricing_icon}",
                            style={"height": "25px", "width": "25px", "marginRight": "20px", "verticalAlign": "middle", "display": "inline-block"}
                        ),
                    ], className="price-text", style={"display": "inline-block", "verticalAlign": "middle"})
                ], className="text-xl font-semibold mt-8 text-gray-800"),
                html.H4("How to Use It:", className="text-lg font-semibold mt-4 text-gray-800"),
                html.P([
                    "1. ", html.Span("Select a Size Category: ", className="font-semibold"), "Choose a specific size category to explore its impact on price.",
                    html.Br(),
                    "2. ", html.Span("Specify a Rent Range: ", className="font-semibold"), "Determine a specific Rent range to investigate."
                ], className="text-lg mt-4 text-gray-600")
            ], className="size_description"),
            dcc.Graph(
                id='scatter-figure_rent',
                figure=scatter_fig_rent(combined_df_r),
                className="mt-8"
            ),
        ]),
        html.Div(className="section-container", children=[
            html.Div([
                html.H1([
                    html.Img(src=f"data:image/jpeg;base64,{encoded_image_bar_icon}", style={"height": "50px", "width": "50px", "marginRight": "10px"}),
                    "bar plot between (bedrooms categorized by property type) with the Rent"
                ], className="text-2xl font-semibold mt-6 flex items-center", style={"font-size": "2rem"}),
                html.H3([
                    "The main goal is to understand the relationship between:",
                    html.Br(),
                    html.Span([
                        "property type ",
                        html.Img(
                            src=f"data:image/jpeg;base64,{encoded_image_property_type_icon}",
                            style={"height": "50px", "width": "50px", "marginRight": "10px", "verticalAlign": "middle", "display": "inline-block"}
                        ),
                    ], className="size-text", style={"display": "inline-block", "verticalAlign": "middle"}),
                    html.Br(),
                    "and",
                    html.Br(),
                    html.Span([
                        "the number of bedrooms ",
                        html.Img(
                            src=f"data:image/jpeg;base64,{encoded_image_bedrooms_icon}",
                            style={"height": "50px", "width": "50px", "marginLeft": "10px", "verticalAlign": "middle", "display": "inline-block"}
                        ),
                    ], className="size-text", style={"display": "inline-block", "verticalAlign": "middle"}),
                    html.Br(),
                    "on",
                    html.Br(),
                    html.Span([
                        "the Rent of the house",
                        html.Img(
                            src=f"data:image/jpeg;base64,{encoded_image_pricing_icon}",
                            style={"height": "25px", "width": "25px", "marginLeft": "10px", "verticalAlign": "middle", "display": "inline-block"}
                        ),
                    ], className="Rent-text", style={"display": "inline-block", "verticalAlign": "middle"})
                ], className="text-xl font-semibold mt-8 text-gray-800"),
                html.H4("How to Use It:", className="text-lg font-semibold mt-4 text-gray-800"),
                html.P([
                    "1. ", html.Span("Select a Property Type Category: ", className="font-semibold"), "Choose a specific property type category to explore its impact on Rent.",
                    html.Br(),
                    "2. ", html.Span("Get the Median of the Bar: ", className="font-semibold"), "You can hover over any specific bar to see the exact median Rent for that number of bedrooms and property type."
                ], className="text-lg mt-4 text-gray-600")
            ], className="bedrooms_description"),
            dcc.Graph(
                id='bar-figure',
                figure=bar_plot_rent(combined_df_r),
                className="mt-8"
            ),
        ]),
        # Fifth Section
        html.Div(className="section-container", children=[
            html.Div([
                html.H1([ 
                    html.Span("Top Regions", style={"color": "green"}), 
                    " and ", 
                    html.Span("Bottom Regions", style={"color": "red"}), 
                    " in Rent in Egypt"
                ], className="text-3xl font-semibold mt-4 text-gray-800"),            
                html.H3("Observe the Rent differences across regions in Egypt to assist your search.", className="text-xl font-semibold mt-8 text-gray-800"),
                html.H4("How to Use:", className="text-lg font-semibold mt-4 text-gray-800"),
                html.P([
                    "1. ", html.Span("Observe the Distribution: ", className="font-semibold"), "The height of each point represents the Rent for each region. Notice the differences to gain insights.",
                    html.Br(),
                    "2. ", html.Span("Hover for Details: ", className="font-semibold"), "Hover over any point to see the region name and Rent. Zoom in to focus on specific areas."
                ], className="text-lg mt-4 text-gray-600")
            ]),
            dcc.Graph(
                id='point_regions',
                figure=region_plot_rent(combined_df_r),
                className="mt-8"
            ),
        ]),
    ])

@app.callback(
    Output("page-content", "children"),
    [
        Input("link-general-sell", "n_clicks"),
        Input("link-general-rent", "n_clicks"),
        Input("link-Customize-sell", "n_clicks"),
        Input("link-Customize-rent", "n_clicks"),
        
    ]
)
def display_page(link_general_sell, link_general_rent, link_Customize_sell,link_Customize_rent):
    ctx = dash.callback_context

    if not ctx.triggered:
        return main_page_layout()
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]

        if button_id == "link-Customize-sell":
            return customize_comparison_layout()
        elif button_id == "link-Customize-rent":
            return cus_rent()
        elif button_id == "link-general-rent":
            return rent_price_layout()  # Change to appropriate layout for General Rent Data
        elif button_id == "link-general-sell":
            return main_page_layout()  # Change to appropriate layout for General Sell Data
        else:
            return main_page_layout()
        

# Custom CSS and index string remain unchanged
app.index_string = '''
<!DOCTYPE html>
<html lang="en">
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
        <style>
        body {
            background-color: white;  /* Set page background color to white */
        }
        .section-container {
            padding: 40px 20px;
            margin-bottom: 20px;
            background-color: #f9f9f9;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }
        .stats-section {
            display: flex;
            justify-content: space-around;
            align-items: center;
            background-color: white; /* Black background for the stats section */
            padding: 20px;
        }
        .stat-item {
            text-align: center;
            color: #fff; /* White text color */
            padding: 20px;
            background-color: #222; /* Slightly different background for the items */
            border-radius: 10px; /* Rounded corners */
            box-shadow: 0 4px 8px rgba(0,0,0,0.1); /* Subtle shadow */
        }
        .stat-item-img {
            display: block;
            margin: 0 auto; /* Center the image horizontally */
        }
        .stat-item .number {
            font-size: 2em;
            font-weight: bold; /* Make the text bold */
            margin: 0;
        }
        .stat-item .label {
            font-size: 1em;
            font-weight: bold; /* Make the text bold */
            margin: 0;
        }
        .jumbotron {
            text-align: center;
            margin-top: 20px;
        }
        .text-4xl {
            font-size: 2.25rem; /* Custom font size */
        }
        .font-bold {
            font-weight: bold; /* Bold text */
        }
        .text-gray-800 {
            color: #1f2937; /* Dark gray text color */
        }
        .text-lg {
            font-size: 1.125rem; /* Large text size */
        }
        .mt-4 {
            margin-top: 1rem; /* Margin top */
        }
        .mt-8 {
            margin-top: 2rem; /* Margin top */
        }
        .text-gray-600 {
            color: #4b5563; /* Medium gray text color */
        }
        .size-text {
            color: #079DD9; /* Green color for size */
            font-weight: bold; /* Ensure bold font for size */
        }
        .price-text {
            color: #D3860A; /* Orange color for price */
            font-weight: bold; /* Ensure bold font for price */
        }
        .relative:hover .absolute {
            display: block;
        }
        .dropdown-content {
            display: none;
            position: absolute;
            background-color: #1f2937;
            min-width: 160px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            z-index: 1;
        }
        .dropdown-content a {
            color: white;
            padding: 12px 16px;
            text-decoration: none;
            display: block;
        }
        .dropdown-content a:hover {
            background-color: #ddd;
            color: black;
        }
        </style>
    </head>
    <body class=".bg-soft-yellow">
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
        <script>
            document.addEventListener('DOMContentLoaded', function() {
                var dropdown = document.querySelector('.group');
                var dropdownContent = dropdown.querySelector('.dropdown-content');

                dropdown.addEventListener('mouseenter', function() {
                    dropdownContent.style.display = 'block';
                });

                dropdown.addEventListener('mouseleave', function() {
                    setTimeout(function() {
                        if (!dropdownContent.matches(':hover')) {
                            dropdownContent.style.display = 'none';
                        }
                    }, 1000); // Delay before hiding the dropdown
                });

                dropdownContent.addEventListener('mouseenter', function() {
                    dropdownContent.style.display = 'block';
                });

                dropdownContent.addEventListener('mouseleave', function() {
                    setTimeout(function() {
                        dropdownContent.style.display = 'none';
                    }, 1000); // Delay before hiding the dropdown
                });
            });
        </script>
    </body>
</html>
'''


# Run the app and generate a link
if __name__ == '__main__':
    app.run_server(debug=True)
