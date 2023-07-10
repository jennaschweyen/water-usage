import matplotlib.pyplot as plt
import pandas as pd
import pickle
import streamlit as st
import mpld3
import streamlit.components.v1 as components

import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.cluster import DBSCAN
from sklearn import metrics

from PIL import Image

st.set_option('deprecation.showPyplotGlobalUse', False)

st.set_page_config(
    page_icon=':ocean:',
    page_title = 'Water Usage',
    initial_sidebar_state='expanded'
)

col1, col2 = st.columns(2)
with col1:
    st.image("https://www.fostercity.org/sites/default/files/styles/gallery500/public/imageattachments/publicworks/page/6071/water-smart.png?itok=vRWn03Zw", 
             width=320)
with col2:
    st.title('Visualizing Water Usage in an Evolving Climate')
st.subheader(''' 
    Bringing water usage discourse closer to home with accessible, contextualized, locally relevant information.
    ''')

page = st.sidebar.selectbox(
    'Page',
    ('About', 'Exploratory Data Analysis', 'Time Series', 'Interactive Maps', 'Cluster Charts', 'Data Frame')
)

if page == 'About':
    st.subheader('About this project')
    st.write('''
Climate change has emerged as a pressing global challenge, with significant implications for water resources.
As temperatures rise and climates become increasingly erratic and unpredictable, the task of monitoring and
managing water usage grows more complex. By leveraging machine learning, we can unlock valuable insights in
relation to water consumption patterns that empower policymakers, water resource managers, and communities to
make informed decisions, develop adaptation strategies, and implement proactive measures to sustainably manage
our water resources in the face of an uncertain climate future.

The objective of this project is to use machine learning to build a clustering model to better understand, and be
able to compare and contrast, state-county level water supply and consumption. By providing locally-relevant
information, consumers of this information might better understand their own water consumption, identifying areas
for improvement and efficiency, and industries in their local area may adjust consumption patterns through awareness 
and advocacy.
             
To complete this analysis, several KMeans clustering unsupervised models were built with the aim that consumers at 
the micro-level (i.e., individuals) and macro-level (e.g., policymakers, water resource managers, agricultural 
authorities, environmental agencies, water conservation groups, etc.) will explore and examine counties of like 
consumption patterns to their own, and subsequently gauge the feasibility and sustainability of those patterns. 
DBScan was also tested as an alternative to KMeans that was robust to outliers, but was not further modeled for 
purposes of computational efficiency and after a predetermination of the desired number of clusters was realized. 
Multi-year temperature and drought data was also modeled using time-series analysis to gain an understanding of 
long-term climate patterns, abrupt climate-related events, trends and variability in temperature and drought conditions, 
and potential insights into future climate scenarios.
             
Our analysis and applications showed that there is widespread variation in water supply and consumption in many
different areas (e.g., industrial, livestock, aquaculture, mining, irrigation, thermoelectric), as well as in 
county-level temperature and drought changes. Our models showed that sufficient similarities exist between counties 
to enable clustering and categorization of counties. Furthermore, engaging storytelling is possible with publicly 
available data. By surfacing these insights and trends with individuals and organizations across the U.S. we may be 
able to shift the conversation around water usage, while considering current climate conditions and trends.
             
We have created these visualizations, models, and dashboards using numerical data from the following sources:

    ''')
    st.markdown("- [Geographical data](https://www.weather.gov/gis/Counties)")
    st.markdown("- [Estimated water usage data](https://www.sciencebase.gov/catalog/item/get/5af3311be4b0da30c1b245d8)")
    st.markdown("- [Temperature data](https://www.nature.com/articles/s41597-022-01405-3/tables/4)")
    st.markdown("- [Drought data](https://droughtmonitor.unl.edu/DmData/DataDownload/ComprehensiveStatistics.aspx)")
    st.markdown("- [Income data](https://data.world/tylerudite/2015-median-income-by-county)")



elif page == 'Exploratory Data Analysis':

    st.subheader('Exploratory Data Analysis')
    st.write('''
We began our analysis with a high-level exploration of the combined data file, including describing all 
numeric variables, visualizing the distribution of water withdrawal and consumption at various levels of 
granularity (e.g., irrigation [crops vs. golf fields], livestock, aquaculture, mining, thermoelectric 
[once through vs. recirculating]),and looking at correlations between different temperatures and drought 
conditions.
             ''')
    
    # EDA completed and images created by Andrew Seefeldt
    image1 = Image.open('../02_EDA/images/Water_Usage_by_Cat.png')
    st.image(image1, 
             caption=' ', 
             width=750,
             channels="RGB", 
             output_format="auto")
    
    st.write('''
Following our preliminary investigation, we explored county level correlations in our combined dataset using 
Shapely files. Each column was mapped individually onto it's own graphic, but no notable insights were observed 
beyond common knowledge/assumptions. For example, we observed that the North is colder than the South, Napa Valley 
draws more water for irrigation than most areas, large cities use larger amounts of drinking water, and the 
southwest have more drought days than the rest of the country.
             ''')
    
    image2 = Image.open('../02_EDA/images/moderate_drought.png')
    st.image(image2, 
             caption=' ', 
             width=750,
             channels="RGB", 
             output_format="auto")
    
    image3 = Image.open('../02_EDA/images/tmean_c.png')
    st.image(image3, 
             caption=' ', 
             width=750,
             channels="RGB", 
             output_format="auto")

    image4 = Image.open('../02_EDA/images/median_household_income.png')
    st.image(image4, 
             caption=' ', 
             width=750,
             channels="RGB", 
             output_format="auto")
    
    
elif page == 'Time Series':

    st.header("County-level Temp and Drought Time Series")
    st.markdown('''
                Climate and Drought Condition data were resampled at a monthly scale using mean values.
                ''')
    st.sidebar.title("Time Series Trend Selector")
    st.sidebar.markdown("Select the charts accordingly:")
    # st.sidebar.checkbox("Show Analysis by County", True, key=1)

    #get the state and county selected in the selectbox
    state = st.sidebar.selectbox('Select your state', ('AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE',
                                'DC', 'FL', 'GA', 'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA',
                                'ME', 'MD', 'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 
                                'NJ', 'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 
                                'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY'))
    if state == 'AL':
        county = st.sidebar.selectbox('Select your county', ('Autauga', 'Baldwin', 'Barbour', 'Bibb', 'Blount',
                                'Bullock', 'Butler', 'Calhoun', 'Chambers', 'Cherokee', 'Chilton',
                                'Choctaw', 'Clarke', 'Clay', 'Cleburne', 'Coffee', 'Colbert', 'Conecuh',
                                'Coosa', 'Covington', 'Crenshaw', 'Cullman', 'Dale', 'Dallas', 'DeKalb', 
                                'Elmore', 'Escambia', 'Etowah', 'Fayette', 'Franklin', 'Geneva', 'Greene',
                                'Hale', 'Henry', 'Houston', 'Jackson', 'Jefferson', 'Lamar', 'Lauderdale',
                                'Lawrence', 'Lee', 'Limestone', 'Lowndes', 'Macon', 'Madison', 'Marengo',
                                'Marion', 'Marshall', 'Mobile', 'Monroe', 'Montgomery', 'Morgan', 'Perry',
                                'Pickens', 'Pike', 'Randolph', 'Russell', 'Shelby', 'St. Clair', 'Sumter', 
                                'Talladega', 'Tallapoosa', 'Tuscaloosa', 'Walker', 'Washington', 'Wilcox', 
                                'Winston'))

    elif state == 'AK':
        county = st.sidebar.selectbox('Select your county', ('Aleutians East', 'Aleutians West', 'Anchorage',
                                'Bethel', 'Bristol Bay', 'Chugach', 'Copper River', 'Denali', 'Dillingham',
                                'Fairbanks North Star', 'Haines', 'Hoonah-Angoon', 'Juneau', 'Kenai Peninsula',
                                'Ketchikan Gateway', 'Kodiak Island', 'Kusilvak', 'Lake and Peninsula',
                                'Matanuska-Susitna', 'Nome', 'North Slope', 'Northwest Arctic', 'Petersburg Borough',
                                'Prince of Wales-Hyder', 'Sitka', 'Skagway', 'Southeast Fairbanks', 'Wrangell',
                                'Yakutat', 'Yukon-Koyukuk'))

    elif state == 'AZ':
        county = st.sidebar.selectbox('Select your county', ('Apache', 'Cochise', 'Coconino', 'Gila', 'Graham',
                                'Greenlee', 'La Paz', 'Maricopa', 'Mohave', 'Navajo', 'Pima', 'Pinal',
                                'Santa Cruz', 'Yavapai', 'Yuma'))

    elif state == 'CA':
        county = st.sidebar.selectbox('Select your county', ('Alameda', 'Alpine', 'Amador', 'Butte', 'Calaveras',
                                'Colusa', 'Contra Costa', 'Del Norte', 'El Dorado', 'Fresno', 'Glenn',
                                'Humboldt', 'Imperial', 'Inyo', 'Kern', 'Kings', 'Lake', 'Lassen',
                                'Los Angeles', 'Madera', 'Marin', 'Mariposa', 'Mendocino', 'Merced', 'Modoc',
                                'Mono', 'Monterey', 'Napa', 'Nevada', 'Orange', 'Placer', 'Plumas',
                                'Riverside', 'Sacramento', 'San Benito', 'San Bernardino', 'San Diego',
                                'San Francisco', 'San Joaquin', 'San Luis Obispo', 'San Mateo', 'Santa Barbara',
                                'Santa Clara', 'Santa Cruz', 'Shasta', 'Sierra', 'Siskiyou', 'Solano', 'Sonoma',
                                'Stanislaus', 'Sutter', 'Tehama', 'Trinity', 'Tulare', 'Tuolumne', 'Ventura',
                                'Yolo', 'Yuba'))

    elif state == 'CO':
        county = st.sidebar.selectbox('Select your county', ('Adams', 'Alamosa', 'Arapahoe', 'Archuleta', 'Baca', 'Bent',
                                'Boulder', 'Broomfield', 'Chaffee', 'Cheyenne', 'Clear Creek', 'Conejos',
                                'Costilla', 'Crowley', 'Custer', 'Delta', 'Denver', 'Dolores', 'Douglas', 'Eagle',
                                'El Paso', 'Elbert', 'Fremont', 'Garfield', 'Gilpin', 'Grand', 'Gunnison',
                                'Hinsdale', 'Huerfano', 'Jackson', 'Jefferson', 'Kiowa', 'Kit Carson', 'La Plata',
                                'Lake', 'Larimer', 'Las Animas', 'Lincoln', 'Logan', 'Mesa', 'Mineral', 'Moffat',
                                'Montezuma', 'Montrose', 'Morgan', 'Otero', 'Ouray', 'Park', 'Phillips', 'Pitkin',
                                'Prowers', 'Pueblo', 'Rio Blanco', 'Rio Grande', 'Routt', 'Saguache', 'San Juan',
                                'San Miguel', 'Sedgwick', 'Summit', 'Teller', 'Washington', 'Weld', 'Yuma'))

    elif state == 'CT':
        county = st.sidebar.selectbox('Select your county', ('Fairfield', 'Hartford', 'Litchfield', 'Middlesex',
                                'New Haven', 'New London', 'Tolland', 'Windham'))

    elif state == 'DE':
        county = st.sidebar.selectbox('Select your county', ('Kent', 'New Castle', 'Sussex'))

    elif state == 'DC':
        county = st.sidebar.selectbox('Select your county', ('District of Columbia'))

    elif state == 'FL':
        county = st.sidebar.selectbox('Select your county', ('Alachua', 'Baker', 'Bay', 'Bradford', 'Brevard', 'Broward',
                                'Calhoun', 'Charlotte', 'Charlotte', 'Citrus', 'Citrus', 'Clay', 'Collier',
                                'Columbia', 'DeSoto', 'Dixie', 'Duval', 'Escambia', 'Flagler', 'Franklin',
                                'Franklin', 'Gadsden', 'Gilchrist', 'Glades', 'Gulf', 'Hamilton', 'Hardee', 'Hendry',
                                'Hernando', 'Highlands', 'Hillsborough', 'Hillsborough', 'Holmes', 'Indian River',
                                'Jackson', 'Jefferson', 'Lafayette', 'Lake', 'Lee', 'Leon', 'Levy', 'Liberty',
                                'Lower Keys in Monroe', 'Madison', 'Mainland Monroe', 'Manatee', 'Marion', 'Martin', 
                                'Miami-Dade', 'Middle Keys in Monroe', 'Nassau', 'Okaloosa', 'Okeechobee', 'Orange',
                                'Osceola', 'Palm Beach', 'Pasco', 'Pinellas', 'Pinellas', 'Polk', 'Putnam', 'Santa Rosa',
                                'Sarasota', 'Sarasota', 'Seminole', 'St. Johns', 'St. Lucie', 'Sumter', 'Suwannee',
                                'Taylor', 'Union', 'Upper Keys in Monroe', 'Volusia', 'Wakulla', 'Walton', 'Washington'))

    elif state == 'GA':
        county = st.sidebar.selectbox('Select your county', ('Appling', 'Atkinson', 'Bacon', 'Baker', 'Baldwin', 'Banks',
                                'Barrow', 'Bartow', 'Ben Hill', 'Berrien', 'Bibb', 'Bleckley', 'Brantley', 'Brooks',
                                'Bryan', 'Bulloch', 'Burke', 'Butts', 'Calhoun', 'Camden', 'Candler', 'Carroll',
                                'Catoosa', 'Charlton', 'Chatham', 'Chattahoochee', 'Chattooga', 'Cherokee', 'Clarke',
                                'Clay', 'Clayton', 'Clinch', 'Cobb', 'Coffee', 'Colquitt', 'Columbia', 'Cook',
                                'Coweta', 'Crawford', 'Crisp', 'Dade', 'Dawson', 'DeKalb', 'Decatur', 'Dodge', 'Dooly',
                                'Dougherty', 'Douglas', 'Early', 'Echols', 'Effingham', 'Elbert', 'Emanuel', 'Evans',
                                'Fannin', 'Fayette', 'Floyd', 'Forsyth', 'Franklin', 'Fulton', 'Gilmer', 'Glascock',
                                'Glynn', 'Gordon', 'Grady', 'Greene', 'Gwinnett', 'Habersham', 'Hall', 'Hancock',
                                'Haralson', 'Harris', 'Hart', 'Heard', 'Henry', 'Houston', 'Irwin', 'Jackson', 'Jasper',
                                'Jeff Davis', 'Jefferson', 'Jenkins', 'Johnson', 'Jones', 'Lamar', 'Lanier', 'Laurens',
                                'Lee', 'Liberty', 'Lincoln', 'Long', 'Lowndes', 'Lumpkin', 'Macon', 'Madison', 'Marion',
                                'McDuffie', 'McIntosh', 'Meriwether', 'Miller', 'Mitchell', 'Monroe', 'Montgomery',
                                'Morgan', 'Murray', 'Muscogee', 'Newton', 'Oconee', 'Oglethorpe', 'Paulding', 'Peach',
                                'Pickens', 'Pierce', 'Pike', 'Polk', 'Pulaski', 'Putnam', 'Quitman', 'Rabun', 'Randolph',
                                'Richmond', 'Rockdale', 'Schley', 'Screven', 'Seminole', 'Spalding', 'Stephens',
                                'Stewart', 'Sumter', 'Talbot', 'Taliaferro', 'Tattnall', 'Taylor', 'Telfair', 'Terrell',
                                'Thomas', 'Tift', 'Toombs', 'Towns', 'Treutlen', 'Troup', 'Turner', 'Twiggs', 'Union',
                                'Upson', 'Walker', 'Walton', 'Ware', 'Warren', 'Washington', 'Wayne', 'Webster', 'Wheeler',
                                'White', 'Whitfield', 'Wilcox', 'Wilkes', 'Wilkinson', 'Worth'))

    elif state == 'HI':
        county = st.sidebar.selectbox('Select your county', ('Hawaii', 'Kahoolawe', 'Kauai', 'Lanai', 'Maui', 'Molokai', 'Niihau', 'Oahu'))

    elif state == 'ID':
        county = st.sidebar.selectbox('Select your county', ('Ada', 'Adams', 'Bannock', 'Bear Lake', 'Benewah', 'Bingham',
                                'Blaine', 'Boise', 'Bonner', 'Bonneville', 'Boundary', 'Butte', 'Camas',
                                'Canyon', 'Caribou', 'Cassia', 'Clark', 'Clearwater', 'Custer', 'Elmore', 'Franklin',
                                'Fremont', 'Gem', 'Gooding', 'Idaho', 'Jefferson', 'Jerome', 'Kootenai',
                                'Latah', 'Lemhi', 'Lewis', 'Lincoln', 'Madison', 'Minidoka', 'Nez Perce',
                                'Oneida', 'Owyhee', 'Payette', 'Power', 'Shoshone', 'Teton', 'Twin Falls',
                                'Valley', 'Washington'))

    elif state == 'IL':
        county = st.sidebar.selectbox('Select your county', ('Adams', 'Alexander', 'Bond', 'Boone', 'Brown', 'Bureau',
                                'Calhoun', 'Carroll', 'Cass', 'Champaign', 'Christian', 'Clark', 'Clay',
                                'Clinton', 'Coles', 'Cook', 'Crawford', 'Cumberland', 'De Kalb', 'De Witt',
                                'Douglas', 'DuPage', 'Edgar', 'Edwards', 'Effingham', 'Fayette', 'Ford',
                                'Franklin', 'Fulton', 'Gallatin', 'Greene', 'Grundy', 'Hamilton', 'Hancock',
                                'Hardin', 'Henderson', 'Henry', 'Iroquois', 'Jackson', 'Jasper', 'Jefferson',
                                'Jersey', 'Jo Daviess', 'Johnson', 'Kane', 'Kankakee', 'Kendall', 'Knox',
                                'La Salle', 'Lake', 'Lawrence', 'Lee', 'Livingston', 'Logan', 'Macon',
                                'Macoupin', 'Madison', 'Marion', 'Marshall', 'Mason', 'Massac', 'McDonough',
                                'McHenry', 'McLean', 'Menard', 'Mercer', 'Monroe', 'Montgomery', 'Morgan',
                                'Moultrie', 'Ogle', 'Peoria', 'Perry', 'Piatt', 'Pike', 'Pope', 'Pulaski',
                                'Putnam', 'Randolph', 'Richland', 'Rock Island', 'Saline', 'Sangamon',
                                'Schuyler', 'Scott', 'Shelby', 'St. Clair', 'Stark', 'Stephenson', 'Tazewell',
                                'Union', 'Vermilion', 'Wabash', 'Warren', 'Washington', 'Wayne', 'White',
                                'Whiteside', 'Will', 'Williamson', 'Winnebago', 'Woodford'))

    elif state == 'IN':
        county = st.sidebar.selectbox('Select your county', ('Adams', 'Allen', 'Bartholomew', 'Benton', 'Blackford',
                                'Boone', 'Brown', 'Carroll', 'Cass', 'Clark', 'Clay', 'Clinton', 'Crawford',
                                'Daviess', 'De Kalb', 'Dearborn', 'Decatur', 'Delaware', 'Dubois', 'Elkhart',
                                'Fayette', 'Floyd', 'Fountain', 'Franklin', 'Fulton', 'Gibson', 'Grant', 'Greene',
                                'Hamilton', 'Hancock', 'Harrison', 'Hendricks', 'Henry', 'Howard', 'Huntington',
                                'Jackson', 'Jasper', 'Jay', 'Jefferson', 'Jennings', 'Johnson', 'Knox',
                                'Kosciusko', 'La Porte', 'Lagrange', 'Lake', 'Lawrence', 'Madison', 'Marion',
                                'Marshall', 'Martin', 'Miami', 'Monroe', 'Montgomery', 'Morgan', 'Newton', 'Noble',
                                'Ohio', 'Orange', 'Owen', 'Parke', 'Perry', 'Pike', 'Porter', 'Posey', 'Pulaski',
                                'Putnam', 'Randolph', 'Ripley', 'Rush', 'Scott', 'Shelby', 'Spencer', 'St. Joseph',
                                'Starke', 'Steuben', 'Sullivan', 'Switzerland', 'Tippecanoe', 'Tipton', 'Union',
                                'Vanderburgh', 'Vermillion', 'Vigo', 'Wabash', 'Warren', 'Warrick', 'Washington',
                                'Wayne', 'Wells', 'White', 'Whitley'))

    elif state == 'IA':
        county = st.sidebar.selectbox('Select your county', ('Adair', 'Adams', 'Allamakee', 'Appanoose', 'Audubon', 'Benton',
                                'Black Hawk', 'Boone', 'Bremer', 'Buchanan', 'Buena Vista', 'Butler', 'Calhoun',
                                'Carroll', 'Cass', 'Cedar', 'Cerro Gordo', 'Cherokee', 'Chickasaw', 'Clarke', 'Clay',
                                'Clayton', 'Clinton', 'Crawford', 'Dallas', 'Davis', 'Decatur', 'Delaware', 'Des Moines',
                                'Dickinson', 'Dubuque', 'Emmet', 'Fayette', 'Floyd', 'Franklin', 'Fremont', 'Greene', 'Grundy',
                                'Guthrie', 'Hamilton', 'Hancock', 'Hardin', 'Harrison', 'Henry', 'Howard', 'Humboldt', 'Ida',
                                'Iowa', 'Jackson', 'Jasper', 'Jefferson', 'Johnson', 'Jones', 'Keokuk', 'Kossuth', 'Lee', 'Linn',
                                'Louisa', 'Lucas', 'Lyon', 'Madison', 'Mahaska', 'Marion', 'Marshall', 'Mills', 'Mitchell', 
                                'Monona', 'Monroe', 'Montgomery', 'Muscatine', "O'Brien", 'Osceola', 'Page', 'Palo Alto',
                                'Plymouth', 'Pocahontas', 'Polk', 'Pottawattamie', 'Poweshiek', 'Ringgold', 'Sac', 'Scott',
                                'Shelby', 'Sioux', 'Story', 'Tama', 'Taylor', 'Union', 'Van Buren', 'Wapello', 'Warren',
                                'Washington', 'Wayne', 'Webster', 'Winnebago', 'Winneshiek', 'Woodbury', 'Worth', 'Wright'))

    elif state == 'KS':
        county = st.sidebar.selectbox('Select your county', ('Allen', 'Anderson', 'Atchison', 'Barber', 'Barton', 'Bourbon', 'Brown',
                                'Butler', 'Chase', 'Chautauqua', 'Cherokee', 'Cheyenne', 'Clark', 'Clay', 'Cloud', 'Coffey',
                                'Comanche', 'Cowley', 'Crawford', 'Decatur', 'Dickinson', 'Doniphan', 'Douglas', 'Edwards', 'Elk', 
                                'Ellis', 'Ellsworth', 'Finney', 'Ford', 'Franklin', 'Geary', 'Gove', 'Graham', 'Grant', 'Gray', 
                                'Greeley', 'Greenwood', 'Hamilton', 'Harper', 'Harvey', 'Haskell', 'Hodgeman', 'Jackson', 'Jefferson',
                                'Jewell', 'Johnson', 'Kearny', 'Kingman', 'Kiowa', 'Labette', 'Lane', 'Leavenworth', 'Lincoln', 'Linn', 
                                'Logan', 'Lyon', 'Marion', 'Marshall', 'McPherson', 'Meade', 'Miami', 'Mitchell', 'Montgomery', 'Morris', 
                                'Morton', 'Nemaha', 'Neosho', 'Ness', 'Norton', 'Osage', 'Osborne', 'Ottawa', 'Pawnee', 'Phillips', 
                                'Pottawatomie', 'Pratt', 'Rawlins', 'Reno', 'Republic', 'Rice', 'Riley', 'Rooks', 'Rush', 'Russell', 
                                'Saline', 'Scott', 'Sedgwick', 'Seward', 'Shawnee', 'Sheridan', 'Sherman', 'Smith', 'Stafford', 'Stanton', 
                                'Stevens', 'Sumner', 'Thomas', 'Trego', 'Wabaunsee', 'Wallace', 'Washington', 'Wichita', 'Wilson', 
                                'Woodson', 'Wyandotte'))

    elif state == 'KY':
        county = st.sidebar.selectbox('Select your county', ('Adair', 'Allen', 'Anderson', 'Ballard', 'Barren', 'Bath', 'Bell', 'Boone', 
                                'Bourbon', 'Boyd', 'Boyle', 'Bracken', 'Breathitt', 'Breckinridge', 'Bullitt', 'Butler', 'Caldwell', 
                                'Calloway', 'Campbell', 'Carlisle', 'Carroll', 'Carter', 'Casey','Christian', 'Clark', 'Clay', 'Clinton', 
                                'Crittenden', 'Cumberland', 'Daviess', 'Edmonson', 'Elliott', 'Estill', 'Fayette', 'Fleming', 'Floyd', 
                                'Franklin', 'Fulton', 'Gallatin', 'Garrard', 'Grant', 'Graves', 'Grayson', 'Green', 'Greenup', 'Hancock', 
                                'Hardin', 'Harlan', 'Harrison', 'Hart', 'Henderson', 'Henry', 'Hickman', 'Hopkins', 'Jackson', 'Jefferson', 
                                'Jessamine', 'Johnson', 'Kenton', 'Knott', 'Knox', 'Larue', 'Laurel', 'Lawrence', 'Lee', 'Leslie', 
                                'Letcher', 'Lewis', 'Lincoln', 'Livingston', 'Logan', 'Lyon', 'Madison', 'Magoffin', 'Marion', 'Marshall', 
                                'Martin', 'Mason', 'McCracken', 'McCreary', 'McLean', 'Meade', 'Menifee', 'Mercer', 'Metcalfe', 'Monroe',
                                'Montgomery', 'Morgan', 'Muhlenberg', 'Nelson', 'Nicholas', 'Ohio', 'Oldham', 'Owen', 'Owsley', 'Pendleton', 
                                'Perry', 'Pike', 'Powell', 'Pulaski', 'Robertson', 'Rockcastle', 'Rowan', 'Russell', 'Scott', 'Shelby', 
                                'Simpson', 'Spencer', 'Taylor', 'Todd', 'Trigg', 'Trimble', 'Union', 'Warren', 'Washington', 'Wayne', 
                                'Webster', 'Whitley', 'Wolfe', 'Woodford'))

    elif state == 'LA':
        county = st.sidebar.selectbox('Select your county', ('Acadia', 'Allen', 'Ascension', 'Assumption', 'Avoyelles', 'Beauregard', 'Bienville', 
                                'Bossier', 'Caddo', 'Calcasieu', 'Caldwell', 'Cameron', 'Catahoula', 'Claiborne', 'Concordia', 'De Soto', 
                                'East Baton Rouge', 'East Carroll', 'East Feliciana', 'Evangeline', 'Franklin', 'Grant', 'Iberia', 
                                'Iberville', 'Jackson', 'Jefferson', 'Jefferson Davis', 'La Salle', 'Lafayette', 'Lafourche', 'Lincoln', 
                                'Livingston', 'Madison', 'Morehouse', 'Natchitoches', 'Orleans', 'Ouachita', 'Plaquemines', 'Pointe Coupee', 
                                'Rapides', 'Red River', 'Richland', 'Sabine', 'St. Bernard', 'St. Charles', 'St. Helena', 'St. James',
                                'St. John The Baptist', 'St. Landry', 'St. Martin', 'St. Mary', 'St. Tammany', 'Tangipahoa', 'Tensas', 
                                'Terrebonne', 'Union', 'Vermilion', 'Vernon', 'Washington', 'Webster', 'West Baton Rouge', 'West Carroll', 
                                'West Feliciana', 'Winn'))

    elif state == 'ME':
        county = st.sidebar.selectbox('Select your county', ('Androscoggin', 'Aroostook', 'Cumberland', 'Franklin', 'Hancock', 'Kennebec', 
                                'Knox', 'Lincoln', 'Oxford', 'Penobscot', 'Piscataquis', 'Sagadahoc', 'Somerset', 'Waldo', 'Washington',
                                'York'))

    elif state == 'MD':
        county = st.sidebar.selectbox('Select your county', ('Allegany', 'Anne Arundel', 'Baltimore', 'Baltimore City', 'Calvert', 'Caroline', 
                                'Carroll', 'Cecil', 'Charles', 'Dorchester', 'Frederick', 'Garrett', 'Harford', 'Howard', 'Kent', 
                                'Montgomery', 'Prince Georges', "Queen Anne's", 'Somerset', 'St. Marys', 'Talbot', 'Washington', 
                                'Wicomico', 'Worcester'))

    elif state == 'MA':
        county = st.sidebar.selectbox('Select your county', ('Barnstable', 'Berkshire', 'Bristol', 'Dukes', 'Essex', 'Franklin', 'Hampden', 
                                'Hampshire', 'Middlesex', 'Nantucket', 'Norfolk', 'Plymouth', 'Suffolk', 'Worcester'))

    elif state == 'MI':
        county = st.sidebar.selectbox('Select your county', ('Alcona', 'Alger', 'Allegan', 'Alpena', 'Antrim', 'Arenac', 'Baraga', 'Barry', 
                                'Bay', 'Benzie', 'Berrien', 'Branch', 'Calhoun', 'Cass', 'Charlevoix', 'Cheboygan', 'Chippewa', 'Clare', 
                                'Clinton', 'Crawford', 'Delta', 'Dickinson', 'Eaton', 'Emmet', 'Genesee', 'Gladwin', 'Gogebic', 
                                'Grand Traverse', 'Gratiot', 'Hillsdale', 'Houghton', 'Huron', 'Ingham', 'Ionia', 'Iosco', 'Iron', 
                                'Isabella', 'Jackson', 'Kalamazoo', 'Kalkaska', 'Kent', 'Keweenaw', 'Lake', 'Lapeer', 'Leelanau', 
                                'Lenawee', 'Livingston', 'Luce', 'Mackinac', 'Macomb', 'Manistee', 'Marquette', 'Mason', 'Mecosta', 
                                'Menominee', 'Midland', 'Missaukee', 'Monroe', 'Montcalm', 'Montmorency', 'Muskegon', 'Newaygo', 'Oakland', 
                                'Oceana', 'Ogemaw', 'Ontonagon', 'Osceola', 'Oscoda', 'Otsego', 'Ottawa', 'Presque Isle', 'Roscommon', 
                                'Saginaw', 'Sanilac', 'Schoolcraft', 'Shiawassee', 'St. Clair', 'St. Joseph', 'Tuscola', 'Van Buren', 
                                'Washtenaw', 'Wayne', 'Wexford'))

    elif state == 'MN':
        county = st.sidebar.selectbox('Select your county', ('Aitkin', 'Anoka', 'Becker', 'Beltrami', 'Benton', 'Big Stone', 'Blue Earth', 'Brown', 
                                'Carlton', 'Carver', 'Cass', 'Chippewa', 'Chisago', 'Clay', 'Clearwater', 'Cook', 'Cottonwood', 'Crow Wing', 
                                'Dakota', 'Dodge', 'Douglas', 'Faribault', 'Fillmore', 'Freeborn', 'Goodhue', 'Grant', 'Hennepin', 'Houston', 
                                'Hubbard', 'Isanti', 'Itasca', 'Jackson', 'Kanabec', 'Kandiyohi', 'Kittson', 'Koochiching', 'Lac qui Parle', 
                                'Lake', 'Lake of the Woods', 'Le Sueur', 'Lincoln', 'Lyon', 'Mahnomen', 'Marshall', 'Martin', 'McLeod', 
                                'Meeker', 'Mille Lacs', 'Morrison', 'Mower', 'Murray', 'Nicollet', 'Nobles', 'Norman', 'Olmsted', 
                                'Otter Tail', 'Pennington', 'Pine', 'Pipestone', 'Polk', 'Pope', 'Ramsey', 'Red Lake', 'Redwood', 
                                'Renville', 'Rice', 'Rock', 'Roseau', 'Scott', 'Sherburne', 'Sibley', 'St. Louis', 'Stearns', 'Steele', 
                                'Stevens', 'Swift', 'Todd', 'Traverse', 'Wabasha', 'Wadena', 'Waseca', 'Washington', 'Watonwan', 'Wilkin', 
                                'Winona', 'Wright', 'Yellow Medicine'))

    elif state == 'MS':
        county = st.sidebar.selectbox('Select your county', ('Adams', 'Alcorn', 'Amite', 'Attala', 'Benton', 'Bolivar', 'Calhoun', 'Carroll', 
                                'Chickasaw', 'Choctaw', 'Claiborne', 'Clarke', 'Clay', 'Coahoma', 'Copiah', 'Covington', 'DeSoto', 
                                'Forrest', 'Franklin', 'George', 'Greene', 'Grenada', 'Hancock', 'Harrison', 'Hinds', 'Holmes', 'Humphreys', 
                                'Issaquena', 'Itawamba', 'Jackson', 'Jasper', 'Jefferson', 'Jefferson Davis', 'Jones', 'Kemper', 'Lafayette', 
                                'Lamar', 'Lauderdale', 'Lawrence', 'Leake', 'Lee', 'Leflore', 'Lincoln', 'Lowndes', 'Madison', 'Marion', 
                                'Marshall', 'Monroe', 'Montgomery', 'Neshoba', 'Newton', 'Noxubee', 'Oktibbeha', 'Panola', 'Pearl River', 
                                'Perry', 'Pike', 'Pontotoc', 'Prentiss', 'Quitman', 'Rankin', 'Scott', 'Sharkey', 'Simpson', 'Smith', 
                                'Stone', 'Sunflower', 'Tallahatchie', 'Tate', 'Tippah', 'Tishomingo', 'Tunica', 'Union', 'Walthall', 
                                'Warren', 'Washington', 'Wayne', 'Webster', 'Wilkinson', 'Winston', 'Yalobusha', 'Yazoo'))

    elif state == 'MO':
        county = st.sidebar.selectbox('Select your county', ('Adair', 'Andrew', 'Atchison', 'Audrain', 'Barry', 'Barton', 
                                'Bates', 'Benton', 'Bollinger', 'Boone', 'Buchanan', 'Butler', 'Caldwell', 'Callaway', 
                                'Camden', 'Cape Girardeau', 'Carroll', 'Carter', 'Cass', 'Cedar', 'Chariton', 'Christian', 
                                'Clark', 'Clay', 'Clinton', 'Cole', 'Cooper', 'Crawford', 'Dade', 'Dallas', 'Daviess', 'DeKalb', 
                                'Dent', 'Douglas', 'Dunklin', 'Franklin', 'Gasconade', 'Gentry', 'Greene', 'Grundy', 'Harrison', 
                                'Henry', 'Hickory', 'Holt', 'Howard', 'Howell', 'Iron', 'Jackson', 'Jasper', 'Jefferson', 'Johnson', 
                                'Knox', 'Laclede', 'Lafayette', 'Lawrence', 'Lewis', 'Lincoln', 'Linn', 'Livingston', 'Macon', 
                                'Madison', 'Maries', 'Marion', 'McDonald', 'Mercer', 'Miller', 'Mississippi', 'Moniteau', 'Monroe', 
                                'Montgomery', 'Morgan', 'New Madrid', 'Newton', 'Nodaway', 'Oregon', 'Osage', 'Ozark', 'Pemiscot', 
                                'Perry', 'Pettis', 'Phelps', 'Pike', 'Platte', 'Polk', 'Pulaski', 'Putnam', 'Ralls', 'Randolph', 'Ray', 
                                'Reynolds', 'Ripley', 'Saline', 'Schuyler', 'Scotland', 'Scott', 'Shannon', 'Shelby', 'St. Charles', 
                                'St. Clair', 'St. Francois', 'St. Louis', 'St. Louis City', 'Ste. Genevieve', 'Stoddard', 'Stone', 
                                'Sullivan', 'Taney', 'Texas', 'Vernon', 'Warren', 'Washington', 'Wayne', 'Webster', 'Worth', 'Wright'))

    elif state == 'MT':
        county = st.sidebar.selectbox('Select your county', ('Beaverhead', 'Big Horn', 'Blaine', 'Broadwater', 'Carbon', 'Carter', 'Cascade', 
                                'Chouteau', 'Custer', 'Daniels', 'Dawson', 'Deer Lodge', 'Fallon', 'Fergus', 'Flathead', 'Gallatin', 
                                'Garfield', 'Glacier', 'Golden Valley', 'Granite', 'Hill', 'Jefferson', 'Judith Basin', 'Lake', 
                                'Lewis and Clark', 'Liberty', 'Lincoln', 'Madison', 'McCone', 'Meagher', 'Mineral', 'Missoula', 
                                'Musselshell', 'Park', 'Petroleum', 'Phillips', 'Pondera', 'Powder River', 'Powell', 'Prairie', 
                                'Ravalli', 'Richland', 'Roosevelt', 'Rosebud', 'Sanders', 'Sheridan', 'Silver Bow', 'Stillwater', 
                                'Sweet Grass', 'Teton', 'Toole', 'Treasure', 'Valley', 'Wheatland', 'Wibaux', 'Yellowstone'))

    elif state == 'NE':
        county = st.sidebar.selectbox('Select your county', ('Adams', 'Antelope', 'Arthur', 'Banner', 'Blaine', 'Boone', 'Box Butte', 'Boyd', 
                                'Brown', 'Buffalo', 'Burt', 'Butler', 'Cass', 'Cedar', 'Chase', 'Cherry', 'Cheyenne', 'Clay', 'Colfax', 
                                'Cuming', 'Custer', 'Dakota', 'Dawes', 'Dawson', 'Deuel', 'Dixon', 'Dodge', 'Douglas', 'Dundy', 'Fillmore', 
                                'Franklin', 'Frontier', 'Furnas', 'Gage', 'Garden', 'Garfield', 'Gosper', 'Grant', 'Greeley', 'Hall', 
                                'Hamilton', 'Harlan', 'Hayes', 'Hitchcock', 'Holt', 'Hooker', 'Howard', 'Jefferson', 'Johnson', 'Kearney', 
                                'Keith', 'Keya Paha', 'Kimball', 'Knox', 'Lancaster', 'Lincoln', 'Logan', 'Loup', 'Madison', 'McPherson', 
                                'Merrick', 'Morrill', 'Nance', 'Nemaha', 'Nuckolls', 'Otoe', 'Pawnee', 'Perkins', 'Phelps', 'Pierce',
                                'Platte', 'Polk', 'Red Willow', 'Richardson', 'Rock', 'Saline', 'Sarpy', 'Saunders', 'Scotts Bluff', 
                                'Seward', 'Sheridan', 'Sherman', 'Sioux', 'Stanton', 'Thayer', 'Thomas', 'Thurston', 'Valley', 
                                'Washington', 'Wayne', 'Webster', 'Wheeler', 'York'))

    elif state == 'NV':
        county = st.sidebar.selectbox('Select your county', ('Carson City', 'Churchill', 'Clark', 'Douglas', 'Elko', 'Esmeralda', 'Eureka', 
                                'Humboldt', 'Lander', 'Lincoln', 'Lyon', 'Mineral', 'Nye', 'Pershing', 'Storey', 'Washoe', 'White Pine'))

    elif state == 'NH':
        county = st.sidebar.selectbox('Select your county', ('Belknap', 'Carroll', 'Cheshire', 'Coos', 'Grafton', 'Hillsborough', 'Merrimack', 
                                'Rockingham', 'Strafford', 'Sullivan'))

    elif state == 'NJ':
        county = st.sidebar.selectbox('Select your county', ('Atlantic', 'Bergen', 'Burlington', 'Camden', 'Cape May', 'Cumberland', 'Essex', 
                                'Gloucester', 'Hudson', 'Hunterdon', 'Mercer', 'Middlesex', 'Monmouth', 'Morris', 'Ocean', 'Passaic', 
                                'Salem', 'Somerset', 'Sussex', 'Union', 'Warren'))

    elif state == 'NM':
        county = st.selectbox('Select your county', ('Bernalillo', 'Catron', 'Chaves', 'Cibola', 'Colfax', 'Curry', 'De Baca', 'Dona Ana', 
                                'Eddy', 'Grant', 'Guadalupe', 'Harding', 'Hidalgo', 'Lea', 'Lincoln', 'Los Alamos', 'Luna', 'McKinley', 
                                'Mora', 'Otero', 'Quay', 'Rio Arriba', 'Roosevelt', 'San Juan', 'San Miguel', 'Sandoval', 'Santa Fe', 
                                'Sierra', 'Socorro', 'Taos', 'Torrance', 'Union', 'Valencia'))

    elif state == 'NY':
        county = st.sidebar.selectbox('Select your county', ('Albany', 'Allegany', 'Bronx', 'Broome', 'Cattaraugus', 'Cayuga', 'Chautauqua', 
                                'Chemung', 'Chenango', 'Clinton', 'Columbia', 'Cortland', 'Delaware', 'Dutchess', 'Erie', 'Essex', 
                                'Franklin', 'Fulton', 'Genesee', 'Greene', 'Hamilton', 'Herkimer', 'Jefferson', 'Kings', 'Lewis', 
                                'Livingston', 'Madison', 'Monroe', 'Montgomery', 'Nassau', 'New York', 'New York (Manhattan)', 
                                'Niagara', 'Oneida', 'Onondaga', 'Ontario', 'Orange', 'Orleans', 'Oswego', 'Otsego', 'Putnam', 'Queens', 
                                'Rensselaer', 'Richmond', 'Rockland', 'Saratoga', 'Schenectady', 'Schoharie', 'Schuyler', 'Seneca', 
                                'St. Lawrence', 'Steuben', 'Suffolk', 'Sullivan', 'Tioga', 'Tompkins', 'Ulster', 'Warren', 'Washington', 
                                'Wayne', 'Westchester', 'Westchester', 'Wyoming', 'Yates'))

    elif state == 'NC':
        county = st.sidebar.selectbox('Select your county', ('Alamance', 'Alexander', 'Alleghany', 'Anson', 'Ashe', 'Avery', 'Beaufort', 'Bertie', 
                                'Bladen', 'Brunswick', 'Buncombe', 'Burke', 'Cabarrus', 'Caldwell', 'Camden', 'Carteret', 'Caswell', 
                                'Catawba', 'Chatham', 'Cherokee', 'Chowan', 'Clay', 'Cleveland', 'Columbus', 'Craven', 'Cumberland', 
                                'Currituck', 'Dare', 'Davidson', 'Davie', 'Duplin', 'Durham', 'Edgecombe', 'Forsyth', 'Franklin', 'Gaston', 
                                'Gates', 'Graham', 'Granville', 'Greene', 'Guilford', 'Halifax', 'Harnett', 'Haywood', 'Henderson', 
                                'Hertford', 'Hoke', 'Hyde', 'Iredell', 'Jackson', 'Johnston', 'Jones', 'Lee', 'Lenoir', 'Lincoln', 'Macon', 
                                'Madison', 'Martin', 'McDowell', 'Mecklenburg', 'Mitchell', 'Montgomery', 'Moore', 'Nash', 'New Hanover', 
                                'Northampton', 'Onslow', 'Orange', 'Pamlico', 'Pasquotank', 'Pender', 'Perquimans', 'Person', 'Pitt', 
                                'Polk', 'Randolph', 'Richmond', 'Robeson', 'Rockingham', 'Rowan', 'Rutherford', 'Sampson', 'Scotland', 
                                'Stanly', 'Stokes', 'Surry', 'Swain', 'Transylvania', 'Tyrrell', 'Union', 'Vance', 'Wake', 'Warren', 
                                'Washington', 'Watauga', 'Wayne', 'Wilkes', 'Wilson', 'Yadkin', 'Yancey'))

    elif state == 'ND':
        county = st.sidebar.selectbox('Select your county', ('Adams', 'Barnes', 'Benson', 'Billings', 'Bottineau', 'Bowman', 'Burke', 'Burleigh', 
                                'Cass', 'Cavalier', 'Dickey', 'Divide', 'Dunn', 'Eddy', 'Emmons', 'Foster', 'Golden Valley', 'Grand Forks', 
                                'Grant', 'Griggs', 'Hettinger', 'Kidder', 'LaMoure', 'Logan', 'McHenry', 'McIntosh', 'McKenzie', 'McLean', 
                                'Mercer', 'Morton', 'Mountrail', 'Nelson', 'Oliver', 'Pembina', 'Pierce', 'Ramsey', 'Ransom', 'Renville', 
                                'Richland', 'Rolette', 'Sargent', 'Sheridan', 'Sioux', 'Slope', 'Stark', 'Steele', 'Stutsman', 'Towner', 
                                'Traill', 'Walsh', 'Ward', 'Wells', 'Williams'))

    elif state == 'OH':
        county = st.sidebar.selectbox('Select your county', ('Adams', 'Allen', 'Ashland', 'Ashtabula', 'Athens', 'Auglaize', 'Belmont', 'Brown', 
                                'Butler', 'Carroll', 'Champaign', 'Clark', 'Clermont', 'Clinton', 'Columbiana', 'Coshocton', 'Crawford', 
                                'Cuyahoga', 'Darke', 'Defiance', 'Delaware', 'Erie', 'Fairfield', 'Fayette', 'Franklin', 'Fulton', 'Gallia', 
                                'Geauga', 'Greene', 'Guernsey', 'Hamilton', 'Hancock', 'Hardin', 'Harrison', 'Henry', 'Highland', 'Hocking', 
                                'Holmes', 'Huron', 'Jackson', 'Jefferson', 'Knox', 'Lake', 'Lawrence', 'Licking', 'Logan', 'Lorain', 
                                'Lucas', 'Madison', 'Mahoning', 'Marion', 'Medina', 'Meigs', 'Mercer', 'Miami', 'Monroe', 'Montgomery', 
                                'Morgan', 'Morrow', 'Muskingum', 'Noble', 'Ottawa', 'Paulding', 'Perry', 'Pickaway', 'Pike', 'Portage', 
                                'Preble', 'Putnam', 'Richland', 'Ross', 'Sandusky', 'Scioto', 'Seneca', 'Shelby', 'Stark', 'Summit', 
                                'Trumbull', 'Tuscarawas', 'Union', 'Van Wert', 'Vinton', 'Warren', 'Washington', 'Wayne', 'Williams', 
                                'Wood', 'Wyandot'))

    elif state == 'OK':
        county = st.sidebar.selectbox('Select your county', ('Adair', 'Alfalfa', 'Atoka', 'Beaver', 'Beckham', 'Blaine', 'Bryan', 'Caddo',
                                'Canadian', 'Carter', 'Cherokee', 'Choctaw', 'Cimarron', 'Cleveland', 'Coal', 'Comanche', 'Cotton',
                                'Craig', 'Creek', 'Custer', 'Delaware', 'Dewey', 'Ellis', 'Garfield', 'Garvin', 'Grady', 'Grant',
                                'Greer', 'Harmon', 'Harper', 'Haskell', 'Hughes', 'Jackson', 'Jefferson', 'Johnston', 'Kay', 'Kingfisher', 
                                'Kiowa', 'Latimer', 'Le Flore', 'Lincoln', 'Logan', 'Love', 'Major', 'Marshall', 'Mayes', 'McClain', 
                                'McCurtain', 'McIntosh', 'Murray', 'Muskogee', 'Noble', 'Nowata', 'Okfuskee', 'Oklahoma', 'Okmulgee', 
                                'Osage', 'Ottawa', 'Pawnee', 'Payne', 'Pittsburg', 'Pontotoc', 'Pottawatomie', 'Pushmataha', 'Roger Mills', 
                                'Rogers', 'Seminole', 'Sequoyah', 'Stephens', 'Texas', 'Tillman', 'Tulsa', 'Wagoner', 'Washington', 
                                'Washita', 'Woods', 'Woodward'))

    elif state == 'OR':
        county = st.sidebar.selectbox('Select your county', ('Baker', 'Benton', 'Clackamas', 'Clatsop', 'Columbia', 'Coos', 'Crook',
                                'Curry', 'Deschutes', 'Douglas', 'Gilliam', 'Grant', 'Harney', 'Hood River', 'Jackson',
                                'Jefferson', 'Josephine', 'Klamath', 'Lake', 'Lane', 'Lincoln', 'Linn', 'Malheur', 'Marion',
                                'Morrow', 'Multnomah', 'Polk', 'Sherman', 'Tillamook', 'Umatilla', 'Union', 'Wallowa', 'Wasco', 
                                'Washington', 'Wheeler', 'Yamhill'))

    elif state == 'PA':
        county = st.sidebar.selectbox('Select your county', ('Adams', 'Allegheny', 'Armstrong', 'Beaver', 'Bedford', 'Berks', 'Blair',
                                'Bradford', 'Bucks', 'Butler', 'Cambria', 'Cameron', 'Carbon', 'Centre', 'Chester', 'Clarion',
                                'Clearfield', 'Clinton', 'Columbia', 'Crawford', 'Cumberland', 'Dauphin', 'Delaware', 'Elk', 'Erie', 
                                'Fayette', 'Forest', 'Franklin', 'Fulton', 'Greene', 'Huntingdon', 'Indiana', 'Jefferson', 'Juniata', 
                                'Lackawanna', 'Lancaster', 'Lawrence', 'Lebanon', 'Lehigh', 'Luzerne', 'Lycoming', 'McKean', 'Mercer',
                                'Mifflin', 'Monroe', 'Montgomery', 'Montour', 'Northampton', 'Northumberland', 'Perry',
                                'Philadelphia', 'Pike', 'Potter', 'Schuylkill', 'Snyder', 'Somerset', 'Sullivan', 'Susquehanna',
                                'Tioga', 'Union', 'Venango', 'Warren', 'Washington', 'Wayne', 'Westmoreland', 'Wyoming', 'York'))

    elif state == 'RI':
        county = st.sidebar.selectbox('Select your county', ('Bristol', 'Kent', 'Newport', 'Providence', 'Washington'))

    elif state == 'SC':
        county = st.sidebar.selectbox('Select your county', ('Abbeville', 'Aiken', 'Allendale', 'Anderson', 'Bamberg',
                                'Barnwell', 'Beaufort', 'Beaufort', 'Berkeley', 'Calhoun', 'Charleston', 
                                'Cherokee', 'Chester', 'Chesterfield', 'Clarendon', 'Colleton', 'Darlington', 
                                'Dillon', 'Dorchester', 'Edgefield', 'Fairfield', 'Florence', 'Georgetown',
                                'Greenville', 'Greenwood', 'Hampton', 'Horry', 'Jasper', 'Kershaw', 'Lancaster',
                                'Laurens', 'Lee', 'Lexington', 'Marion', 'Marlboro', 'McCormick', 'Newberry',
                                'Oconee', 'Orangeburg', 'Pickens', 'Richland', 'Saluda', 'Spartanburg', 'Sumter',
                                'Union', 'Williamsburg', 'York'))

    elif state == 'SD':
        county = st.sidebar.selectbox('Select your county', ('Aurora', 'Beadle', 'Bennett', 'Bon Homme', 'Brookings', 'Brown',
                                'Brule', 'Buffalo', 'Butte', 'Campbell', 'Charles Mix', 'Clark', 'Clay', 'Codington',
                                'Corson', 'Custer', 'Davison', 'Day', 'Deuel', 'Dewey', 'Douglas', 'Edmunds',
                                'Fall River', 'Faulk', 'Grant', 'Gregory', 'Haakon', 'Hamlin', 'Hand', 'Hanson',
                                'Harding', 'Hughes', 'Hutchinson', 'Hyde', 'Jackson', 'Jerauld', 'Jones', 'Kingsbury',
                                'Lake', 'Lawrence', 'Lincoln', 'Lyman', 'Marshall', 'McCook', 'McPherson', 'Meade',
                                'Mellette', 'Miner', 'Minnehaha', 'Moody', 'Oglala Lakota', 'Pennington', 'Perkins',
                                'Potter', 'Roberts', 'Sanborn', 'Spink', 'Stanley', 'Sully', 'Todd', 'Tripp', 'Turner',
                                'Union', 'Walworth', 'Yankton', 'Ziebach'))

    elif state == 'TN':
        county = st.sidebar.selectbox('Select your county', ('Anderson', 'Bedford', 'Benton', 'Bledsoe', 'Blount', 'Bradley', 
                                'Campbell', 'Cannon', 'Carroll', 'Carter', 'Cheatham', 'Chester', 'Claiborne', 'Clay',
                                'Cocke', 'Coffee', 'Crockett', 'Cumberland', 'Davidson', 'De Kalb', 'Decatur', 'Dickson',
                                'Dyer', 'Fayette', 'Fentress', 'Franklin', 'Gibson', 'Giles', 'Grainger', 'Greene', 'Grundy',
                                'Hamblen', 'Hamilton', 'Hancock', 'Hardeman', 'Hardin', 'Hawkins', 'Haywood', 'Henderson',
                                'Henry', 'Hickman', 'Houston', 'Humphreys', 'Jackson', 'Jefferson', 'Johnson', 'Knox', 'Lake',
                                'Lauderdale', 'Lawrence', 'Lewis', 'Lincoln', 'Loudon', 'Macon', 'Madison', 'Marion', 'Marshall',
                                'Maury', 'McMinn', 'McNairy', 'Meigs', 'Monroe', 'Montgomery', 'Moore', 'Morgan', 'Obion',
                                'Overton', 'Perry', 'Pickett', 'Polk', 'Putnam', 'Rhea', 'Roane', 'Robertson', 'Rutherford',
                                'Scott', 'Sequatchie', 'Sevier', 'Shelby', 'Smith', 'Stewart', 'Sullivan', 'Sumner', 'Tipton',
                                'Trousdale', 'Unicoi', 'Union', 'Van Buren', 'Warren', 'Washington', 'Wayne', 'Weakley',
                                'White', 'Williamson', 'Wilson'))

    elif state == 'TX':
        county = st.sidebar.selectbox('Select your county', ('Anderson', 'Andrews', 'Angelina', 'Aransas', 'Archer', 'Armstrong',
                                'Atascosa', 'Austin', 'Bailey', 'Bandera', 'Bastrop', 'Baylor', 'Bee', 'Bell', 'Bexar',
                                'Blanco', 'Borden', 'Bosque', 'Bowie', 'Brazoria', 'Brazos', 'Brewster', 'Briscoe',
                                'Brooks', 'Brown', 'Burleson', 'Burnet', 'Caldwell', 'Calhoun', 'Callahan', 'Cameron', 
                                'Camp', 'Carson', 'Cass', 'Castro', 'Chambers', 'Cherokee', 'Childress', 'Clay', 'Cochran', 
                                'Coke', 'Coleman', 'Collin', 'Collingsworth', 'Colorado', 'Comal', 'Comanche', 'Concho', 
                                'Cooke', 'Coryell', 'Cottle', 'Crane', 'Crockett', 'Crosby', 'Culberson', 'Dallam', 'Dallas', 
                                'Dawson', 'DeWitt', 'Deaf Smith', 'Delta', 'Denton', 'Dickens', 'Dimmit', 'Donley', 'Duval', 
                                'Eastland', 'Ector', 'Edwards', 'El Paso', 'Ellis', 'Erath', 'Falls', 'Fannin', 'Fayette',
                                'Fisher', 'Floyd', 'Foard', 'Fort Bend', 'Franklin', 'Freestone', 'Frio', 'Gaines', 'Galveston',
                                'Garza', 'Gillespie', 'Glasscock', 'Goliad', 'Gonzales', 'Gray', 'Grayson', 'Gregg', 'Grimes',
                                'Guadalupe', 'Hale', 'Hall', 'Hamilton', 'Hansford', 'Hardeman', 'Hardin', 'Harris', 'Harrison',
                                'Hartley', 'Haskell', 'Hays', 'Hemphill', 'Henderson', 'Hidalgo', 'Hill', 'Hockley', 'Hood',
                                'Hopkins', 'Houston', 'Howard', 'Hudspeth', 'Hunt', 'Hutchinson', 'Irion', 'Jack', 'Jackson',
                                'Jasper', 'Jeff Davis', 'Jefferson', 'Jim Hogg', 'Jim Wells', 'Johnson', 'Jones', 'Karnes',
                                'Kaufman', 'Kendall', 'Kenedy', 'Kent', 'Kerr', 'Kimble', 'King', 'Kinney', 'Kleberg', 'Knox',
                                'La Salle', 'Lamar', 'Lamb', 'Lampasas', 'Lavaca', 'Lee', 'Leon', 'Liberty', 'Limestone', 'Lipscomb', 
                                'Live Oak', 'Llano', 'Loving', 'Lubbock', 'Lynn', 'Madison', 'Marion', 'Martin', 'Mason', 'Matagorda', 
                                'Maverick', 'McCulloch', 'McLennan', 'McMullen', 'Medina', 'Menard', 'Midland', 'Milam', 'Mills',
                                'Mitchell', 'Montague', 'Montgomery', 'Moore', 'Morris', 'Motley', 'Nacogdoches', 'Navarro', 'Newton',
                                'Nolan', 'Nueces', 'Ochiltree', 'Oldham', 'Orange', 'Palo Pinto', 'Panola', 'Parker', 'Parmer',
                                'Pecos', 'Polk', 'Potter', 'Presidio', 'Rains', 'Randall', 'Reagan', 'Real', 'Red River', 'Reeves', 
                                'Refugio', 'Roberts', 'Robertson', 'Rockwall', 'Runnels', 'Rusk', 'Sabine', 'San Augustine',
                                'San Jacinto', 'San Patricio', 'San Saba', 'Schleicher', 'Scurry', 'Shackelford', 'Shelby', 'Sherman', 
                                'Smith', 'Somervell', 'Starr', 'Stephens', 'Sterling', 'Stonewall', 'Sutton', 'Swisher', 'Tarrant',
                                'Taylor', 'Terrell', 'Terry', 'Throckmorton', 'Titus', 'Tom Green', 'Travis', 'Trinity', 'Tyler',
                                'Upshur', 'Upton', 'Uvalde', 'Val Verde', 'Van Zandt', 'Victoria', 'Walker', 'Waller', 'Ward',
                                'Washington', 'Webb', 'Wharton', 'Wheeler', 'Wichita', 'Wilbarger', 'Willacy', 'Williamson', 'Wilson', 
                                'Winkler', 'Wise', 'Wood', 'Yoakum', 'Young', 'Zapata', 'Zavala'))

    elif state == 'UT':
        county = st.sidebar.selectbox('Select your county', ('Beaver', 'Box Elder', 'Cache', 'Carbon', 'Daggett',
                                'Davis', 'Duchesne', 'Emery', 'Garfield', 'Grand', 'Iron', 'Juab',
                                'Kane', 'Millard', 'Morgan', 'Piute', 'Rich', 'Salt Lake', 'San Juan',
                                'Sanpete', 'Sevier', 'Summit', 'Tooele', 'Uintah', 'Utah', 'Wasatch',
                                'Washington', 'Wayne', 'Weber'))

    elif state == 'VT':
        county = st.sidebar.selectbox('Select your county', ('Addison', 'Bennington', 'Caledonia', 'Chittenden', 
                                'Essex', 'Franklin', 'Grand Isle', 'Lamoille', 'Orange', 'Orleans', 'Rutland',
                                'Washington', 'Windham', 'Windsor'))

    elif state == 'VA':
        county = st.sidebar.selectbox('Select your county', ('Accomack', 'Albemarle', 'Alleghany', 'Amelia', 'Amherst', 
                                'Appomattox', 'Arlington', 'Augusta', 'Bath', 'Bedford', 'Bland', 'Botetourt',
                                'Brunswick', 'Buchanan', 'Buckingham', 'Campbell', 'Caroline', 'Carroll',
                                'Charles City', 'Charlotte', 'Chesterfield', 'Clarke', 'Craig', 'Culpeper',
                                'Cumberland', 'Dickenson', 'Dinwiddie', 'Essex', 'Fairfax', 'Fauquier', 'Floyd',
                                'Fluvanna', 'Franklin', 'Frederick', 'Giles', 'Gloucester', 'Goochland', 'Grayson',
                                'Greene', 'Greensville', 'Halifax', 'Hanover', 'Henrico', 'Henry', 'Highland',
                                'Isle of Wight', 'James City', 'King George', 'King William', 'King and Queen',
                                'Lancaster', 'Lee', 'Loudoun', 'Louisa', 'Lunenburg', 'Madison', 'Mathews',
                                'Mecklenburg', 'Middlesex', 'Montgomery', 'Nelson', 'New Kent', 'Northampton',
                                'Northumberland', 'Nottoway', 'Orange', 'Page', 'Patrick', 'Pittsylvania',
                                'Powhatan', 'Prince Edward', 'Prince George', 'Prince William', 'Pulaski',
                                'Rappahannock', 'Richmond', 'Roanoke', 'Rockbridge', 'Rockingham', 'Russell',
                                'Scott', 'Shenandoah', 'Smyth', 'Southampton', 'Spotsylvania', 'Stafford', 'Surry',
                                'Sussex', 'Tazewell', 'Warren', 'Washington', 'Westmoreland', 'Wise', 'Wythe', 'York'))

    elif state == 'WA':
        county = st.sidebar.selectbox('Select your county', ('Adams', 'Asotin', 'Benton', 'Chelan', 'Clallam',
                                'Clark', 'Columbia', 'Cowlitz', 'Douglas', 'Ferry', 'Franklin', 'Garfield',
                                'Grant', 'Grays Harbor', 'Island', 'Jefferson', 'King', 'Kitsap',
                                'Kittitas', 'Klickitat', 'Lewis', 'Lincoln', 'Mason', 'Okanogan', 'Pacific',
                                'Pend Oreille', 'Pierce', 'San Juan', 'Skagit', 'Skamania', 'Snohomish',
                                'Spokane', 'Stevens', 'Thurston', 'Wahkiakum', 'Walla Walla', 'Whatcom',
                                'Whitman', 'Yakima'
        ))

    elif state == 'WV':
        county = st.sidebar.selectbox('Select your county', ('Barbour', 'Berkeley', 'Boone', 'Braxton', 'Brooke',
                                'Cabell', 'Calhoun', 'Clay', 'Doddridge', 'Fayette', 'Gilmer', 'Grant',
                                'Greenbrier', 'Hampshire', 'Hancock', 'Hardy', 'Harrison', 'Jackson',
                                'Jefferson', 'Kanawha', 'Lewis', 'Lincoln', 'Logan', 'Marion', 'Marshall',
                                'Mason', 'McDowell', 'Mercer', 'Mineral', 'Mingo', 'Monongalia', 'Monroe',
                                'Morgan', 'Nicholas', 'Ohio', 'Pendleton', 'Pleasants', 'Pocahontas',
                                'Preston', 'Putnam', 'Raleigh', 'Randolph', 'Ritchie', 'Roane', 'Summers',
                                'Taylor', 'Tucker', 'Tyler', 'Upshur', 'Wayne', 'Webster', 'Wetzel', 'Wirt',
                                'Wood', 'Wyoming'))

    elif state == 'WI':
        county = st.sidebar.selectbox('Select your county', ('Adams', 'Ashland', 'Barron', 'Bayfield', 'Brown',
                                'Buffalo', 'Burnett', 'Calumet', 'Chippewa', 'Clark', 'Columbia', 'Crawford', 
                                'Dane', 'Dodge', 'Door', 'Douglas', 'Dunn', 'Eau Claire', 'Florence', 
                                'Fond du Lac', 'Forest', 'Grant', 'Green', 'Green Lake', 'Iowa', 'Iron',
                                'Jackson', 'Jefferson', 'Juneau', 'Kenosha', 'Kewaunee', 'La Crosse',
                                'Lafayette', 'Langlade', 'Lincoln', 'Manitowoc', 'Marathon', 'Marinette',
                                'Marquette', 'Menominee', 'Milwaukee', 'Monroe', 'Oconto', 'Oneida', 'Outagamie',
                                'Ozaukee', 'Pepin', 'Pierce', 'Polk', 'Portage', 'Price', 'Racine', 'Richland',
                                'Rock', 'Rusk', 'Sauk', 'Sawyer', 'Shawano', 'Sheboygan', 'St. Croix', 'Taylor',
                                'Trempealeau', 'Vernon', 'Vilas', 'Walworth', 'Washburn', 'Washington', 'Waukesha',
                                'Waupaca', 'Waushara', 'Winnebago', 'Wood'))

    elif state == 'WY':
        county = st.sidebar.selectbox('Select your county', ('Albany', 'Big Horn', 'Campbell', 'Carbon', 'Converse', 'Crook',
                                'Fremont', 'Goshen', 'Hot Springs', 'Johnson', 'Laramie', 'Lincoln', 'Natrona', 'Niobrara',
                                'Park', 'Platte', 'Sheridan', 'Sublette', 'Sweetwater', 'Teton', 'Uinta', 'Washakie',
                                'Weston'))
    
    select_status = st.sidebar.radio("Select a time series chart", ('Temperature Trends by County',
                                                    'Drought Trends by County'))
    
    # Convert state and county name to fips code
    df = pd.read_csv('../../data/clean-data/combined2.csv')
    state_data = df[df['state'] == state]
    fips = int(df[(df['state'] == state) & (df['countyname'] == county)]['fips'])
    fips = str(fips)
    fips = fips.zfill(5)

    ## Time Series Citation: Bob Adams
    # Set up data frame for monthly time series
    mon = pd.read_csv('../../data/clean-data/Monthly_Temp_Drought_Combo.csv', dtype = {'FIPS':str})
    mon.drop(columns = 'Unnamed: 0', inplace = True)
    mon.rename(columns = {
        'Month' : 'month',
        'FIPS' : 'fips',
        'Tmin_C' : 'min_temp',
        'Tmax_C' : 'max_temp',
        'Tmean_C' : 'mean_temp',
        'Flag_T' : 'flag_pop_covered'
        }, inplace = True)
    mon['month'] = pd.to_datetime(mon['month'],format = ('%Y-%m'))
    #cite :https://stackoverflow.com/a/339024 for rjust 
    mon['fips'] = mon['fips'].str.rjust(5,'0')
    # Convert Celsius to Farenheit to limit confusion within the U.S. Market
    mon[['min_temp','max_temp','mean_temp']] *= (9/5)
    mon[['min_temp','max_temp','mean_temp']] += 32 

    # Set up data frame for yearly time series
    year = pd.read_csv('../../data/clean-data/Temp_Drought_Combo.csv', dtype= {'FIPS' : str})
    year.drop(columns = 'Unnamed: 0', inplace = True)
    year['year'] = pd.to_datetime(year['year'].astype(str))
    year['FIPS'] = year['FIPS'].str.rjust(5,'0') 

    # Create county dictionary to enable human readable outputs
    counties = pd.read_csv('../../data/raw-data/counties.csv', dtype = {'FIPS': str})
    counties.drop(columns = 'Unnamed: 0', inplace = True)
    counties['FIPS'] = counties['FIPS'].str.rjust(5,'0')
    county_dict = dict(zip(counties['FIPS'], zip(counties['STATE'], counties['COUNTYNAME'], counties['LON'], counties['LAT'])))

    # Local time series plotting functions
    def plot_temp_trends_county(county_fips, min_year, county, state):    

        # Filtered Monthly Summary View
        county_month_view_df = mon[(mon['month'].dt.year >= min_year) & (mon['fips'] == county_fips)]
        county_month_view_df.set_index('month', inplace = True)

        # Annual Summary from Daily Data
        county_year_view_df = year[(year['year'].dt.year >= min_year) & (year['FIPS'] == county_fips)]
        county_year_view_df = county_year_view_df[['year','FIPS','Tmean_C']]
        # Convert to Farenheit
        county_year_view_df['Tmean_C'] *= (9/5)
        county_year_view_df['Tmean_C'] += 32
        county_year_view_df.rename(columns = {'Tmean_C' : 'Tmean_F'}, inplace = True)

        county_year_view_df.set_index('year', inplace = True)
        #cite: Time Series in Pandas Lesson
    
        # Plot
        plt.figure(figsize = (12,8))
        plt.plot(county_month_view_df['min_temp'], c = '#EED78D', label = 'Low Temp (F)')
        plt.plot(county_month_view_df['max_temp'], c = '#C22B26',  label = 'High Temp (F)')
        plt.plot(county_month_view_df['mean_temp'], c = '#FFB632',  label = 'Mean Temp (F)')
        plt.plot(county_year_view_df['Tmean_F'], c = 'k', label = 'Annual Mean Temp (F)',)

        plt.title(f"Temperature Trend for {county} County, {state}")
        plt.yticks(fontsize = 12)
        plt.xticks(fontsize = 12)
        plt.ylabel('Average Monthly Temperature (F)', fontsize = 12)
        plt.legend()
        st.pyplot();   

    def plot_drought_trends_county(county_fips, min_year, county, state):
        # Filtered Monthly Summary View
        county_month_view_df = mon[(mon['month'].dt.year >= min_year) & (mon['fips'] == county_fips)]
        county_month_view_df.set_index('month', inplace = True)
        county_month_view_df['extreme_plus'] = county_month_view_df[['exceptional_drought','extreme_drought']].max(axis = 1)
        county_month_view_df['severe_plus'] = county_month_view_df[['exceptional_drought','extreme_drought','severe_drought']].max(axis = 1)
        county_month_view_df['moderate_plus'] = county_month_view_df[['exceptional_drought','extreme_drought','severe_drought', 'moderate_drought']].max(axis = 1)

        # Annual Summary from Daily Data
        county_year_view_df = year[(year['year'].dt.year >= min_year) & (year['FIPS'] == county_fips)]
        county_year_view_df = county_year_view_df[['year','FIPS','Tmean_C']]
        # Convert to Fahrenheit
        county_year_view_df['Tmean_C'] *= (9/5)
        county_year_view_df['Tmean_C'] += 32
        county_year_view_df.rename(columns = {'Tmean_C' : 'Tmean_F'}, inplace = True)

        county_year_view_df.set_index('year', inplace = True)
        #cite: Time Series in Pandas Lesson
    
        # Plot
        plt.figure(figsize = (12,8))
        plt.plot(county_month_view_df['exceptional_drought'], c = '#C22B26', label = 'Exceptional Drought')
        plt.plot(county_month_view_df['extreme_plus'], c = '#D58900',  label = 'Extreme Drought')
        plt.plot(county_month_view_df['severe_plus'], c = '#FFB632',  label = 'Severe Drought')
        plt.plot(county_month_view_df['moderate_plus'], c = '#EED78D',  label = 'Moderate Drought')

        plt.title(f"Average Minimum Drought Condition for {county} County, {state}")
        plt.yticks(fontsize = 12)
        plt.xticks(fontsize = 12)
        plt.ylabel('Percent Population Experiencing Designated Drought Condition or Worse', fontsize = 12)
        plt.legend()
        st.pyplot();

    min_year = 2010

    if select_status == 'Temperature Trends by County':
        st.markdown("### Temperature Trends by County")
        st.markdown("###### Area charts depicting monthly temperature ranges (min-mean-max) are provided, " +
                    "annual averages and a fixed annual mean temperature, indexed against the first year in " +
                    "the date range. This helps keep track of longer term trends.")
        plot_temp_trends_county(fips, min_year, county, state)

    if select_status == 'Drought Trends by County':
        st.markdown("### Drought Trends by County")
        st.markdown("###### Drought Conditions are visualized similarly using area charts. As the drought " +
                    "categories are sequential (Moderate > Severe > Extreme > Exceptional), values are " +
                    "calculated as the percent of population experiencing at least the specified drought " +
                    "condition. Areas tend to enter and exit drought conditions sequentially.")
        plot_drought_trends_county(fips, min_year, county, state)


elif page == 'Interactive Maps':
    
    st.title('What did we find?')
    # Tableau Dashboard created by Andrew Seefeldt
    # Embedded Tableau Code
    # Cite: https://discuss.streamlit.io/t/how-to-embed-the-tableau-with-iframe-properly/17408
    import streamlit as st
    import streamlit.components.v1 as components

    components.html(
        """
            <div class='tableauPlaceholder' id='viz1688953411063' style='position: relative'><noscript><a href='#'><img alt='Dashboard 1 ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Wa&#47;WaterUsageUSA&#47;Dashboard1&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='path' value='views&#47;WaterUsageUSA&#47;Dashboard1?:language=en-US&amp;:embed=true&amp;publish=yes' /> <param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Wa&#47;WaterUsageUSA&#47;Dashboard1&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /><param name='language' value='en-US' /><param name='filter' value='publish=yes' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1688953411063');                    var vizElement = divElement.getElementsByTagName('object')[0];                    if ( divElement.offsetWidth > 800 ) { vizElement.style.minWidth='520px';vizElement.style.maxWidth='1520px';vizElement.style.width='100%';vizElement.style.minHeight='587px';vizElement.style.maxHeight='887px';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';} else if ( divElement.offsetWidth > 500 ) { vizElement.style.minWidth='520px';vizElement.style.maxWidth='1520px';vizElement.style.width='100%';vizElement.style.minHeight='587px';vizElement.style.maxHeight='887px';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';} else { vizElement.style.width='100%';vizElement.style.height='1427px';}                     var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>
        """,
        width=900,
        height=650,
        scrolling=True
)

 
    # Choropleths
    #cite: https://plotly.com/python/county-choropleth/ > Redirects to manage deprecation: https://plotly.com/python/choropleth-maps/

    from urllib.request import urlopen
    import json
    with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
        counties = json.load(response)

    # dtype={'FIPS':str} # need to make sure we are using FIPS codes as strings vs. ints.  Can be done on import.

    import plotly.express as px

    # Get user input: which cluster would you like to see for your selected county
    # user_selected_value = '' # Get user input: which cluster would you like to see for your selected county
    # cluster_dict = {
    #     '<user_selected_value':'column_name',
    # }

    # fig = px.choropleth(
    #                     title = f"Water Usage for {county}",
    #                     # df, ## dateframe with FIPs codes
    #                     geojson=counties,
    #                     locations='FIPS',
    #                     color=cluster_dict.get(user_selected_value),
    #                     hover_name = 'county', ## County Name
    #                     hover_data = '',
    #                     basemap_visible = True
    #                     )



elif page == 'Cluster Charts':

    # Kmeans Cluster charts created by Farah Malik and Bryan Ortiz
    df = pd.read_csv('../../data/clean-data/combined2.csv')

    st.header("County-level Water Usage Dashboard")
    st.markdown('''
                Clustering can be used to explore connections and uncover correlations 
                that are not easily seen in a two-dimensional map.
                ''')
    st.sidebar.title("Cluster Chart Selector")
    st.sidebar.markdown("Select the charts accordingly:")
    st.sidebar.checkbox("Show Analysis by County", True, key=1)

    #get the state and county selected in the selectbox
    state = st.sidebar.selectbox('Select your state', ('AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE',
                                'DC', 'FL', 'GA', 'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA',
                                'ME', 'MD', 'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 
                                'NJ', 'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 
                                'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY'))
    if state == 'AL':
        county = st.sidebar.selectbox('Select your county', ('Autauga', 'Baldwin', 'Barbour', 'Bibb', 'Blount',
                                'Bullock', 'Butler', 'Calhoun', 'Chambers', 'Cherokee', 'Chilton',
                                'Choctaw', 'Clarke', 'Clay', 'Cleburne', 'Coffee', 'Colbert', 'Conecuh',
                                'Coosa', 'Covington', 'Crenshaw', 'Cullman', 'Dale', 'Dallas', 'DeKalb', 
                                'Elmore', 'Escambia', 'Etowah', 'Fayette', 'Franklin', 'Geneva', 'Greene',
                                'Hale', 'Henry', 'Houston', 'Jackson', 'Jefferson', 'Lamar', 'Lauderdale',
                                'Lawrence', 'Lee', 'Limestone', 'Lowndes', 'Macon', 'Madison', 'Marengo',
                                'Marion', 'Marshall', 'Mobile', 'Monroe', 'Montgomery', 'Morgan', 'Perry',
                                'Pickens', 'Pike', 'Randolph', 'Russell', 'Shelby', 'St. Clair', 'Sumter', 
                                'Talladega', 'Tallapoosa', 'Tuscaloosa', 'Walker', 'Washington', 'Wilcox', 
                                'Winston'))

    elif state == 'AK':
        county = st.sidebar.selectbox('Select your county', ('Aleutians East', 'Aleutians West', 'Anchorage',
                                'Bethel', 'Bristol Bay', 'Chugach', 'Copper River', 'Denali', 'Dillingham',
                                'Fairbanks North Star', 'Haines', 'Hoonah-Angoon', 'Juneau', 'Kenai Peninsula',
                                'Ketchikan Gateway', 'Kodiak Island', 'Kusilvak', 'Lake and Peninsula',
                                'Matanuska-Susitna', 'Nome', 'North Slope', 'Northwest Arctic', 'Petersburg Borough',
                                'Prince of Wales-Hyder', 'Sitka', 'Skagway', 'Southeast Fairbanks', 'Wrangell',
                                'Yakutat', 'Yukon-Koyukuk'))

    elif state == 'AZ':
        county = st.sidebar.selectbox('Select your county', ('Apache', 'Cochise', 'Coconino', 'Gila', 'Graham',
                                'Greenlee', 'La Paz', 'Maricopa', 'Mohave', 'Navajo', 'Pima', 'Pinal',
                                'Santa Cruz', 'Yavapai', 'Yuma'))

    elif state == 'CA':
        county = st.sidebar.selectbox('Select your county', ('Alameda', 'Alpine', 'Amador', 'Butte', 'Calaveras',
                                'Colusa', 'Contra Costa', 'Del Norte', 'El Dorado', 'Fresno', 'Glenn',
                                'Humboldt', 'Imperial', 'Inyo', 'Kern', 'Kings', 'Lake', 'Lassen',
                                'Los Angeles', 'Madera', 'Marin', 'Mariposa', 'Mendocino', 'Merced', 'Modoc',
                                'Mono', 'Monterey', 'Napa', 'Nevada', 'Orange', 'Placer', 'Plumas',
                                'Riverside', 'Sacramento', 'San Benito', 'San Bernardino', 'San Diego',
                                'San Francisco', 'San Joaquin', 'San Luis Obispo', 'San Mateo', 'Santa Barbara',
                                'Santa Clara', 'Santa Cruz', 'Shasta', 'Sierra', 'Siskiyou', 'Solano', 'Sonoma',
                                'Stanislaus', 'Sutter', 'Tehama', 'Trinity', 'Tulare', 'Tuolumne', 'Ventura',
                                'Yolo', 'Yuba'))

    elif state == 'CO':
        county = st.sidebar.selectbox('Select your county', ('Adams', 'Alamosa', 'Arapahoe', 'Archuleta', 'Baca', 'Bent',
                                'Boulder', 'Broomfield', 'Chaffee', 'Cheyenne', 'Clear Creek', 'Conejos',
                                'Costilla', 'Crowley', 'Custer', 'Delta', 'Denver', 'Dolores', 'Douglas', 'Eagle',
                                'El Paso', 'Elbert', 'Fremont', 'Garfield', 'Gilpin', 'Grand', 'Gunnison',
                                'Hinsdale', 'Huerfano', 'Jackson', 'Jefferson', 'Kiowa', 'Kit Carson', 'La Plata',
                                'Lake', 'Larimer', 'Las Animas', 'Lincoln', 'Logan', 'Mesa', 'Mineral', 'Moffat',
                                'Montezuma', 'Montrose', 'Morgan', 'Otero', 'Ouray', 'Park', 'Phillips', 'Pitkin',
                                'Prowers', 'Pueblo', 'Rio Blanco', 'Rio Grande', 'Routt', 'Saguache', 'San Juan',
                                'San Miguel', 'Sedgwick', 'Summit', 'Teller', 'Washington', 'Weld', 'Yuma'))

    elif state == 'CT':
        county = st.sidebar.selectbox('Select your county', ('Fairfield', 'Hartford', 'Litchfield', 'Middlesex',
                                'New Haven', 'New London', 'Tolland', 'Windham'))

    elif state == 'DE':
        county = st.sidebar.selectbox('Select your county', ('Kent', 'New Castle', 'Sussex'))

    elif state == 'DC':
        county = st.sidebar.selectbox('Select your county', ('District of Columbia'))

    elif state == 'FL':
        county = st.sidebar.selectbox('Select your county', ('Alachua', 'Baker', 'Bay', 'Bradford', 'Brevard', 'Broward',
                                'Calhoun', 'Charlotte', 'Charlotte', 'Citrus', 'Citrus', 'Clay', 'Collier',
                                'Columbia', 'DeSoto', 'Dixie', 'Duval', 'Escambia', 'Flagler', 'Franklin',
                                'Franklin', 'Gadsden', 'Gilchrist', 'Glades', 'Gulf', 'Hamilton', 'Hardee', 'Hendry',
                                'Hernando', 'Highlands', 'Hillsborough', 'Hillsborough', 'Holmes', 'Indian River',
                                'Jackson', 'Jefferson', 'Lafayette', 'Lake', 'Lee', 'Leon', 'Levy', 'Liberty',
                                'Lower Keys in Monroe', 'Madison', 'Mainland Monroe', 'Manatee', 'Marion', 'Martin', 
                                'Miami-Dade', 'Middle Keys in Monroe', 'Nassau', 'Okaloosa', 'Okeechobee', 'Orange',
                                'Osceola', 'Palm Beach', 'Pasco', 'Pinellas', 'Pinellas', 'Polk', 'Putnam', 'Santa Rosa',
                                'Sarasota', 'Sarasota', 'Seminole', 'St. Johns', 'St. Lucie', 'Sumter', 'Suwannee',
                                'Taylor', 'Union', 'Upper Keys in Monroe', 'Volusia', 'Wakulla', 'Walton', 'Washington'))

    elif state == 'GA':
        county = st.sidebar.selectbox('Select your county', ('Appling', 'Atkinson', 'Bacon', 'Baker', 'Baldwin', 'Banks',
                                'Barrow', 'Bartow', 'Ben Hill', 'Berrien', 'Bibb', 'Bleckley', 'Brantley', 'Brooks',
                                'Bryan', 'Bulloch', 'Burke', 'Butts', 'Calhoun', 'Camden', 'Candler', 'Carroll',
                                'Catoosa', 'Charlton', 'Chatham', 'Chattahoochee', 'Chattooga', 'Cherokee', 'Clarke',
                                'Clay', 'Clayton', 'Clinch', 'Cobb', 'Coffee', 'Colquitt', 'Columbia', 'Cook',
                                'Coweta', 'Crawford', 'Crisp', 'Dade', 'Dawson', 'DeKalb', 'Decatur', 'Dodge', 'Dooly',
                                'Dougherty', 'Douglas', 'Early', 'Echols', 'Effingham', 'Elbert', 'Emanuel', 'Evans',
                                'Fannin', 'Fayette', 'Floyd', 'Forsyth', 'Franklin', 'Fulton', 'Gilmer', 'Glascock',
                                'Glynn', 'Gordon', 'Grady', 'Greene', 'Gwinnett', 'Habersham', 'Hall', 'Hancock',
                                'Haralson', 'Harris', 'Hart', 'Heard', 'Henry', 'Houston', 'Irwin', 'Jackson', 'Jasper',
                                'Jeff Davis', 'Jefferson', 'Jenkins', 'Johnson', 'Jones', 'Lamar', 'Lanier', 'Laurens',
                                'Lee', 'Liberty', 'Lincoln', 'Long', 'Lowndes', 'Lumpkin', 'Macon', 'Madison', 'Marion',
                                'McDuffie', 'McIntosh', 'Meriwether', 'Miller', 'Mitchell', 'Monroe', 'Montgomery',
                                'Morgan', 'Murray', 'Muscogee', 'Newton', 'Oconee', 'Oglethorpe', 'Paulding', 'Peach',
                                'Pickens', 'Pierce', 'Pike', 'Polk', 'Pulaski', 'Putnam', 'Quitman', 'Rabun', 'Randolph',
                                'Richmond', 'Rockdale', 'Schley', 'Screven', 'Seminole', 'Spalding', 'Stephens',
                                'Stewart', 'Sumter', 'Talbot', 'Taliaferro', 'Tattnall', 'Taylor', 'Telfair', 'Terrell',
                                'Thomas', 'Tift', 'Toombs', 'Towns', 'Treutlen', 'Troup', 'Turner', 'Twiggs', 'Union',
                                'Upson', 'Walker', 'Walton', 'Ware', 'Warren', 'Washington', 'Wayne', 'Webster', 'Wheeler',
                                'White', 'Whitfield', 'Wilcox', 'Wilkes', 'Wilkinson', 'Worth'))

    elif state == 'HI':
        county = st.sidebar.selectbox('Select your county', ('Hawaii', 'Kahoolawe', 'Kauai', 'Lanai', 'Maui', 'Molokai', 'Niihau', 'Oahu'))

    elif state == 'ID':
        county = st.sidebar.selectbox('Select your county', ('Ada', 'Adams', 'Bannock', 'Bear Lake', 'Benewah', 'Bingham',
                                'Blaine', 'Boise', 'Bonner', 'Bonneville', 'Boundary', 'Butte', 'Camas',
                                'Canyon', 'Caribou', 'Cassia', 'Clark', 'Clearwater', 'Custer', 'Elmore', 'Franklin',
                                'Fremont', 'Gem', 'Gooding', 'Idaho', 'Jefferson', 'Jerome', 'Kootenai',
                                'Latah', 'Lemhi', 'Lewis', 'Lincoln', 'Madison', 'Minidoka', 'Nez Perce',
                                'Oneida', 'Owyhee', 'Payette', 'Power', 'Shoshone', 'Teton', 'Twin Falls',
                                'Valley', 'Washington'))

    elif state == 'IL':
        county = st.sidebar.selectbox('Select your county', ('Adams', 'Alexander', 'Bond', 'Boone', 'Brown', 'Bureau',
                                'Calhoun', 'Carroll', 'Cass', 'Champaign', 'Christian', 'Clark', 'Clay',
                                'Clinton', 'Coles', 'Cook', 'Crawford', 'Cumberland', 'De Kalb', 'De Witt',
                                'Douglas', 'DuPage', 'Edgar', 'Edwards', 'Effingham', 'Fayette', 'Ford',
                                'Franklin', 'Fulton', 'Gallatin', 'Greene', 'Grundy', 'Hamilton', 'Hancock',
                                'Hardin', 'Henderson', 'Henry', 'Iroquois', 'Jackson', 'Jasper', 'Jefferson',
                                'Jersey', 'Jo Daviess', 'Johnson', 'Kane', 'Kankakee', 'Kendall', 'Knox',
                                'La Salle', 'Lake', 'Lawrence', 'Lee', 'Livingston', 'Logan', 'Macon',
                                'Macoupin', 'Madison', 'Marion', 'Marshall', 'Mason', 'Massac', 'McDonough',
                                'McHenry', 'McLean', 'Menard', 'Mercer', 'Monroe', 'Montgomery', 'Morgan',
                                'Moultrie', 'Ogle', 'Peoria', 'Perry', 'Piatt', 'Pike', 'Pope', 'Pulaski',
                                'Putnam', 'Randolph', 'Richland', 'Rock Island', 'Saline', 'Sangamon',
                                'Schuyler', 'Scott', 'Shelby', 'St. Clair', 'Stark', 'Stephenson', 'Tazewell',
                                'Union', 'Vermilion', 'Wabash', 'Warren', 'Washington', 'Wayne', 'White',
                                'Whiteside', 'Will', 'Williamson', 'Winnebago', 'Woodford'))

    elif state == 'IN':
        county = st.sidebar.selectbox('Select your county', ('Adams', 'Allen', 'Bartholomew', 'Benton', 'Blackford',
                                'Boone', 'Brown', 'Carroll', 'Cass', 'Clark', 'Clay', 'Clinton', 'Crawford',
                                'Daviess', 'De Kalb', 'Dearborn', 'Decatur', 'Delaware', 'Dubois', 'Elkhart',
                                'Fayette', 'Floyd', 'Fountain', 'Franklin', 'Fulton', 'Gibson', 'Grant', 'Greene',
                                'Hamilton', 'Hancock', 'Harrison', 'Hendricks', 'Henry', 'Howard', 'Huntington',
                                'Jackson', 'Jasper', 'Jay', 'Jefferson', 'Jennings', 'Johnson', 'Knox',
                                'Kosciusko', 'La Porte', 'Lagrange', 'Lake', 'Lawrence', 'Madison', 'Marion',
                                'Marshall', 'Martin', 'Miami', 'Monroe', 'Montgomery', 'Morgan', 'Newton', 'Noble',
                                'Ohio', 'Orange', 'Owen', 'Parke', 'Perry', 'Pike', 'Porter', 'Posey', 'Pulaski',
                                'Putnam', 'Randolph', 'Ripley', 'Rush', 'Scott', 'Shelby', 'Spencer', 'St. Joseph',
                                'Starke', 'Steuben', 'Sullivan', 'Switzerland', 'Tippecanoe', 'Tipton', 'Union',
                                'Vanderburgh', 'Vermillion', 'Vigo', 'Wabash', 'Warren', 'Warrick', 'Washington',
                                'Wayne', 'Wells', 'White', 'Whitley'))

    elif state == 'IA':
        county = st.sidebar.selectbox('Select your county', ('Adair', 'Adams', 'Allamakee', 'Appanoose', 'Audubon', 'Benton',
                                'Black Hawk', 'Boone', 'Bremer', 'Buchanan', 'Buena Vista', 'Butler', 'Calhoun',
                                'Carroll', 'Cass', 'Cedar', 'Cerro Gordo', 'Cherokee', 'Chickasaw', 'Clarke', 'Clay',
                                'Clayton', 'Clinton', 'Crawford', 'Dallas', 'Davis', 'Decatur', 'Delaware', 'Des Moines',
                                'Dickinson', 'Dubuque', 'Emmet', 'Fayette', 'Floyd', 'Franklin', 'Fremont', 'Greene', 'Grundy',
                                'Guthrie', 'Hamilton', 'Hancock', 'Hardin', 'Harrison', 'Henry', 'Howard', 'Humboldt', 'Ida',
                                'Iowa', 'Jackson', 'Jasper', 'Jefferson', 'Johnson', 'Jones', 'Keokuk', 'Kossuth', 'Lee', 'Linn',
                                'Louisa', 'Lucas', 'Lyon', 'Madison', 'Mahaska', 'Marion', 'Marshall', 'Mills', 'Mitchell', 
                                'Monona', 'Monroe', 'Montgomery', 'Muscatine', "O'Brien", 'Osceola', 'Page', 'Palo Alto',
                                'Plymouth', 'Pocahontas', 'Polk', 'Pottawattamie', 'Poweshiek', 'Ringgold', 'Sac', 'Scott',
                                'Shelby', 'Sioux', 'Story', 'Tama', 'Taylor', 'Union', 'Van Buren', 'Wapello', 'Warren',
                                'Washington', 'Wayne', 'Webster', 'Winnebago', 'Winneshiek', 'Woodbury', 'Worth', 'Wright'))

    elif state == 'KS':
        county = st.sidebar.selectbox('Select your county', ('Allen', 'Anderson', 'Atchison', 'Barber', 'Barton', 'Bourbon', 'Brown',
                                'Butler', 'Chase', 'Chautauqua', 'Cherokee', 'Cheyenne', 'Clark', 'Clay', 'Cloud', 'Coffey',
                                'Comanche', 'Cowley', 'Crawford', 'Decatur', 'Dickinson', 'Doniphan', 'Douglas', 'Edwards', 'Elk', 
                                'Ellis', 'Ellsworth', 'Finney', 'Ford', 'Franklin', 'Geary', 'Gove', 'Graham', 'Grant', 'Gray', 
                                'Greeley', 'Greenwood', 'Hamilton', 'Harper', 'Harvey', 'Haskell', 'Hodgeman', 'Jackson', 'Jefferson',
                                'Jewell', 'Johnson', 'Kearny', 'Kingman', 'Kiowa', 'Labette', 'Lane', 'Leavenworth', 'Lincoln', 'Linn', 
                                'Logan', 'Lyon', 'Marion', 'Marshall', 'McPherson', 'Meade', 'Miami', 'Mitchell', 'Montgomery', 'Morris', 
                                'Morton', 'Nemaha', 'Neosho', 'Ness', 'Norton', 'Osage', 'Osborne', 'Ottawa', 'Pawnee', 'Phillips', 
                                'Pottawatomie', 'Pratt', 'Rawlins', 'Reno', 'Republic', 'Rice', 'Riley', 'Rooks', 'Rush', 'Russell', 
                                'Saline', 'Scott', 'Sedgwick', 'Seward', 'Shawnee', 'Sheridan', 'Sherman', 'Smith', 'Stafford', 'Stanton', 
                                'Stevens', 'Sumner', 'Thomas', 'Trego', 'Wabaunsee', 'Wallace', 'Washington', 'Wichita', 'Wilson', 
                                'Woodson', 'Wyandotte'))

    elif state == 'KY':
        county = st.sidebar.selectbox('Select your county', ('Adair', 'Allen', 'Anderson', 'Ballard', 'Barren', 'Bath', 'Bell', 'Boone', 
                                'Bourbon', 'Boyd', 'Boyle', 'Bracken', 'Breathitt', 'Breckinridge', 'Bullitt', 'Butler', 'Caldwell', 
                                'Calloway', 'Campbell', 'Carlisle', 'Carroll', 'Carter', 'Casey','Christian', 'Clark', 'Clay', 'Clinton', 
                                'Crittenden', 'Cumberland', 'Daviess', 'Edmonson', 'Elliott', 'Estill', 'Fayette', 'Fleming', 'Floyd', 
                                'Franklin', 'Fulton', 'Gallatin', 'Garrard', 'Grant', 'Graves', 'Grayson', 'Green', 'Greenup', 'Hancock', 
                                'Hardin', 'Harlan', 'Harrison', 'Hart', 'Henderson', 'Henry', 'Hickman', 'Hopkins', 'Jackson', 'Jefferson', 
                                'Jessamine', 'Johnson', 'Kenton', 'Knott', 'Knox', 'Larue', 'Laurel', 'Lawrence', 'Lee', 'Leslie', 
                                'Letcher', 'Lewis', 'Lincoln', 'Livingston', 'Logan', 'Lyon', 'Madison', 'Magoffin', 'Marion', 'Marshall', 
                                'Martin', 'Mason', 'McCracken', 'McCreary', 'McLean', 'Meade', 'Menifee', 'Mercer', 'Metcalfe', 'Monroe',
                                'Montgomery', 'Morgan', 'Muhlenberg', 'Nelson', 'Nicholas', 'Ohio', 'Oldham', 'Owen', 'Owsley', 'Pendleton', 
                                'Perry', 'Pike', 'Powell', 'Pulaski', 'Robertson', 'Rockcastle', 'Rowan', 'Russell', 'Scott', 'Shelby', 
                                'Simpson', 'Spencer', 'Taylor', 'Todd', 'Trigg', 'Trimble', 'Union', 'Warren', 'Washington', 'Wayne', 
                                'Webster', 'Whitley', 'Wolfe', 'Woodford'))

    elif state == 'LA':
        county = st.sidebar.selectbox('Select your county', ('Acadia', 'Allen', 'Ascension', 'Assumption', 'Avoyelles', 'Beauregard', 'Bienville', 
                                'Bossier', 'Caddo', 'Calcasieu', 'Caldwell', 'Cameron', 'Catahoula', 'Claiborne', 'Concordia', 'De Soto', 
                                'East Baton Rouge', 'East Carroll', 'East Feliciana', 'Evangeline', 'Franklin', 'Grant', 'Iberia', 
                                'Iberville', 'Jackson', 'Jefferson', 'Jefferson Davis', 'La Salle', 'Lafayette', 'Lafourche', 'Lincoln', 
                                'Livingston', 'Madison', 'Morehouse', 'Natchitoches', 'Orleans', 'Ouachita', 'Plaquemines', 'Pointe Coupee', 
                                'Rapides', 'Red River', 'Richland', 'Sabine', 'St. Bernard', 'St. Charles', 'St. Helena', 'St. James',
                                'St. John The Baptist', 'St. Landry', 'St. Martin', 'St. Mary', 'St. Tammany', 'Tangipahoa', 'Tensas', 
                                'Terrebonne', 'Union', 'Vermilion', 'Vernon', 'Washington', 'Webster', 'West Baton Rouge', 'West Carroll', 
                                'West Feliciana', 'Winn'))

    elif state == 'ME':
        county = st.sidebar.selectbox('Select your county', ('Androscoggin', 'Aroostook', 'Cumberland', 'Franklin', 'Hancock', 'Kennebec', 
                                'Knox', 'Lincoln', 'Oxford', 'Penobscot', 'Piscataquis', 'Sagadahoc', 'Somerset', 'Waldo', 'Washington',
                                'York'))

    elif state == 'MD':
        county = st.sidebar.selectbox('Select your county', ('Allegany', 'Anne Arundel', 'Baltimore', 'Baltimore City', 'Calvert', 'Caroline', 
                                'Carroll', 'Cecil', 'Charles', 'Dorchester', 'Frederick', 'Garrett', 'Harford', 'Howard', 'Kent', 
                                'Montgomery', 'Prince Georges', "Queen Anne's", 'Somerset', 'St. Marys', 'Talbot', 'Washington', 
                                'Wicomico', 'Worcester'))

    elif state == 'MA':
        county = st.sidebar.selectbox('Select your county', ('Barnstable', 'Berkshire', 'Bristol', 'Dukes', 'Essex', 'Franklin', 'Hampden', 
                                'Hampshire', 'Middlesex', 'Nantucket', 'Norfolk', 'Plymouth', 'Suffolk', 'Worcester'))

    elif state == 'MI':
        county = st.sidebar.selectbox('Select your county', ('Alcona', 'Alger', 'Allegan', 'Alpena', 'Antrim', 'Arenac', 'Baraga', 'Barry', 
                                'Bay', 'Benzie', 'Berrien', 'Branch', 'Calhoun', 'Cass', 'Charlevoix', 'Cheboygan', 'Chippewa', 'Clare', 
                                'Clinton', 'Crawford', 'Delta', 'Dickinson', 'Eaton', 'Emmet', 'Genesee', 'Gladwin', 'Gogebic', 
                                'Grand Traverse', 'Gratiot', 'Hillsdale', 'Houghton', 'Huron', 'Ingham', 'Ionia', 'Iosco', 'Iron', 
                                'Isabella', 'Jackson', 'Kalamazoo', 'Kalkaska', 'Kent', 'Keweenaw', 'Lake', 'Lapeer', 'Leelanau', 
                                'Lenawee', 'Livingston', 'Luce', 'Mackinac', 'Macomb', 'Manistee', 'Marquette', 'Mason', 'Mecosta', 
                                'Menominee', 'Midland', 'Missaukee', 'Monroe', 'Montcalm', 'Montmorency', 'Muskegon', 'Newaygo', 'Oakland', 
                                'Oceana', 'Ogemaw', 'Ontonagon', 'Osceola', 'Oscoda', 'Otsego', 'Ottawa', 'Presque Isle', 'Roscommon', 
                                'Saginaw', 'Sanilac', 'Schoolcraft', 'Shiawassee', 'St. Clair', 'St. Joseph', 'Tuscola', 'Van Buren', 
                                'Washtenaw', 'Wayne', 'Wexford'))

    elif state == 'MN':
        county = st.sidebar.selectbox('Select your county', ('Aitkin', 'Anoka', 'Becker', 'Beltrami', 'Benton', 'Big Stone', 'Blue Earth', 'Brown', 
                                'Carlton', 'Carver', 'Cass', 'Chippewa', 'Chisago', 'Clay', 'Clearwater', 'Cook', 'Cottonwood', 'Crow Wing', 
                                'Dakota', 'Dodge', 'Douglas', 'Faribault', 'Fillmore', 'Freeborn', 'Goodhue', 'Grant', 'Hennepin', 'Houston', 
                                'Hubbard', 'Isanti', 'Itasca', 'Jackson', 'Kanabec', 'Kandiyohi', 'Kittson', 'Koochiching', 'Lac qui Parle', 
                                'Lake', 'Lake of the Woods', 'Le Sueur', 'Lincoln', 'Lyon', 'Mahnomen', 'Marshall', 'Martin', 'McLeod', 
                                'Meeker', 'Mille Lacs', 'Morrison', 'Mower', 'Murray', 'Nicollet', 'Nobles', 'Norman', 'Olmsted', 
                                'Otter Tail', 'Pennington', 'Pine', 'Pipestone', 'Polk', 'Pope', 'Ramsey', 'Red Lake', 'Redwood', 
                                'Renville', 'Rice', 'Rock', 'Roseau', 'Scott', 'Sherburne', 'Sibley', 'St. Louis', 'Stearns', 'Steele', 
                                'Stevens', 'Swift', 'Todd', 'Traverse', 'Wabasha', 'Wadena', 'Waseca', 'Washington', 'Watonwan', 'Wilkin', 
                                'Winona', 'Wright', 'Yellow Medicine'))

    elif state == 'MS':
        county = st.sidebar.selectbox('Select your county', ('Adams', 'Alcorn', 'Amite', 'Attala', 'Benton', 'Bolivar', 'Calhoun', 'Carroll', 
                                'Chickasaw', 'Choctaw', 'Claiborne', 'Clarke', 'Clay', 'Coahoma', 'Copiah', 'Covington', 'DeSoto', 
                                'Forrest', 'Franklin', 'George', 'Greene', 'Grenada', 'Hancock', 'Harrison', 'Hinds', 'Holmes', 'Humphreys', 
                                'Issaquena', 'Itawamba', 'Jackson', 'Jasper', 'Jefferson', 'Jefferson Davis', 'Jones', 'Kemper', 'Lafayette', 
                                'Lamar', 'Lauderdale', 'Lawrence', 'Leake', 'Lee', 'Leflore', 'Lincoln', 'Lowndes', 'Madison', 'Marion', 
                                'Marshall', 'Monroe', 'Montgomery', 'Neshoba', 'Newton', 'Noxubee', 'Oktibbeha', 'Panola', 'Pearl River', 
                                'Perry', 'Pike', 'Pontotoc', 'Prentiss', 'Quitman', 'Rankin', 'Scott', 'Sharkey', 'Simpson', 'Smith', 
                                'Stone', 'Sunflower', 'Tallahatchie', 'Tate', 'Tippah', 'Tishomingo', 'Tunica', 'Union', 'Walthall', 
                                'Warren', 'Washington', 'Wayne', 'Webster', 'Wilkinson', 'Winston', 'Yalobusha', 'Yazoo'))

    elif state == 'MO':
        county = st.sidebar.selectbox('Select your county', ('Adair', 'Andrew', 'Atchison', 'Audrain', 'Barry', 'Barton', 
                                'Bates', 'Benton', 'Bollinger', 'Boone', 'Buchanan', 'Butler', 'Caldwell', 'Callaway', 
                                'Camden', 'Cape Girardeau', 'Carroll', 'Carter', 'Cass', 'Cedar', 'Chariton', 'Christian', 
                                'Clark', 'Clay', 'Clinton', 'Cole', 'Cooper', 'Crawford', 'Dade', 'Dallas', 'Daviess', 'DeKalb', 
                                'Dent', 'Douglas', 'Dunklin', 'Franklin', 'Gasconade', 'Gentry', 'Greene', 'Grundy', 'Harrison', 
                                'Henry', 'Hickory', 'Holt', 'Howard', 'Howell', 'Iron', 'Jackson', 'Jasper', 'Jefferson', 'Johnson', 
                                'Knox', 'Laclede', 'Lafayette', 'Lawrence', 'Lewis', 'Lincoln', 'Linn', 'Livingston', 'Macon', 
                                'Madison', 'Maries', 'Marion', 'McDonald', 'Mercer', 'Miller', 'Mississippi', 'Moniteau', 'Monroe', 
                                'Montgomery', 'Morgan', 'New Madrid', 'Newton', 'Nodaway', 'Oregon', 'Osage', 'Ozark', 'Pemiscot', 
                                'Perry', 'Pettis', 'Phelps', 'Pike', 'Platte', 'Polk', 'Pulaski', 'Putnam', 'Ralls', 'Randolph', 'Ray', 
                                'Reynolds', 'Ripley', 'Saline', 'Schuyler', 'Scotland', 'Scott', 'Shannon', 'Shelby', 'St. Charles', 
                                'St. Clair', 'St. Francois', 'St. Louis', 'St. Louis City', 'Ste. Genevieve', 'Stoddard', 'Stone', 
                                'Sullivan', 'Taney', 'Texas', 'Vernon', 'Warren', 'Washington', 'Wayne', 'Webster', 'Worth', 'Wright'))

    elif state == 'MT':
        county = st.sidebar.selectbox('Select your county', ('Beaverhead', 'Big Horn', 'Blaine', 'Broadwater', 'Carbon', 'Carter', 'Cascade', 
                                'Chouteau', 'Custer', 'Daniels', 'Dawson', 'Deer Lodge', 'Fallon', 'Fergus', 'Flathead', 'Gallatin', 
                                'Garfield', 'Glacier', 'Golden Valley', 'Granite', 'Hill', 'Jefferson', 'Judith Basin', 'Lake', 
                                'Lewis and Clark', 'Liberty', 'Lincoln', 'Madison', 'McCone', 'Meagher', 'Mineral', 'Missoula', 
                                'Musselshell', 'Park', 'Petroleum', 'Phillips', 'Pondera', 'Powder River', 'Powell', 'Prairie', 
                                'Ravalli', 'Richland', 'Roosevelt', 'Rosebud', 'Sanders', 'Sheridan', 'Silver Bow', 'Stillwater', 
                                'Sweet Grass', 'Teton', 'Toole', 'Treasure', 'Valley', 'Wheatland', 'Wibaux', 'Yellowstone'))

    elif state == 'NE':
        county = st.sidebar.selectbox('Select your county', ('Adams', 'Antelope', 'Arthur', 'Banner', 'Blaine', 'Boone', 'Box Butte', 'Boyd', 
                                'Brown', 'Buffalo', 'Burt', 'Butler', 'Cass', 'Cedar', 'Chase', 'Cherry', 'Cheyenne', 'Clay', 'Colfax', 
                                'Cuming', 'Custer', 'Dakota', 'Dawes', 'Dawson', 'Deuel', 'Dixon', 'Dodge', 'Douglas', 'Dundy', 'Fillmore', 
                                'Franklin', 'Frontier', 'Furnas', 'Gage', 'Garden', 'Garfield', 'Gosper', 'Grant', 'Greeley', 'Hall', 
                                'Hamilton', 'Harlan', 'Hayes', 'Hitchcock', 'Holt', 'Hooker', 'Howard', 'Jefferson', 'Johnson', 'Kearney', 
                                'Keith', 'Keya Paha', 'Kimball', 'Knox', 'Lancaster', 'Lincoln', 'Logan', 'Loup', 'Madison', 'McPherson', 
                                'Merrick', 'Morrill', 'Nance', 'Nemaha', 'Nuckolls', 'Otoe', 'Pawnee', 'Perkins', 'Phelps', 'Pierce',
                                'Platte', 'Polk', 'Red Willow', 'Richardson', 'Rock', 'Saline', 'Sarpy', 'Saunders', 'Scotts Bluff', 
                                'Seward', 'Sheridan', 'Sherman', 'Sioux', 'Stanton', 'Thayer', 'Thomas', 'Thurston', 'Valley', 
                                'Washington', 'Wayne', 'Webster', 'Wheeler', 'York'))

    elif state == 'NV':
        county = st.sidebar.selectbox('Select your county', ('Carson City', 'Churchill', 'Clark', 'Douglas', 'Elko', 'Esmeralda', 'Eureka', 
                                'Humboldt', 'Lander', 'Lincoln', 'Lyon', 'Mineral', 'Nye', 'Pershing', 'Storey', 'Washoe', 'White Pine'))

    elif state == 'NH':
        county = st.sidebar.selectbox('Select your county', ('Belknap', 'Carroll', 'Cheshire', 'Coos', 'Grafton', 'Hillsborough', 'Merrimack', 
                                'Rockingham', 'Strafford', 'Sullivan'))

    elif state == 'NJ':
        county = st.sidebar.selectbox('Select your county', ('Atlantic', 'Bergen', 'Burlington', 'Camden', 'Cape May', 'Cumberland', 'Essex', 
                                'Gloucester', 'Hudson', 'Hunterdon', 'Mercer', 'Middlesex', 'Monmouth', 'Morris', 'Ocean', 'Passaic', 
                                'Salem', 'Somerset', 'Sussex', 'Union', 'Warren'))

    elif state == 'NM':
        county = st.selectbox('Select your county', ('Bernalillo', 'Catron', 'Chaves', 'Cibola', 'Colfax', 'Curry', 'De Baca', 'Dona Ana', 
                                'Eddy', 'Grant', 'Guadalupe', 'Harding', 'Hidalgo', 'Lea', 'Lincoln', 'Los Alamos', 'Luna', 'McKinley', 
                                'Mora', 'Otero', 'Quay', 'Rio Arriba', 'Roosevelt', 'San Juan', 'San Miguel', 'Sandoval', 'Santa Fe', 
                                'Sierra', 'Socorro', 'Taos', 'Torrance', 'Union', 'Valencia'))

    elif state == 'NY':
        county = st.sidebar.selectbox('Select your county', ('Albany', 'Allegany', 'Bronx', 'Broome', 'Cattaraugus', 'Cayuga', 'Chautauqua', 
                                'Chemung', 'Chenango', 'Clinton', 'Columbia', 'Cortland', 'Delaware', 'Dutchess', 'Erie', 'Essex', 
                                'Franklin', 'Fulton', 'Genesee', 'Greene', 'Hamilton', 'Herkimer', 'Jefferson', 'Kings', 'Lewis', 
                                'Livingston', 'Madison', 'Monroe', 'Montgomery', 'Nassau', 'New York', 'New York (Manhattan)', 
                                'Niagara', 'Oneida', 'Onondaga', 'Ontario', 'Orange', 'Orleans', 'Oswego', 'Otsego', 'Putnam', 'Queens', 
                                'Rensselaer', 'Richmond', 'Rockland', 'Saratoga', 'Schenectady', 'Schoharie', 'Schuyler', 'Seneca', 
                                'St. Lawrence', 'Steuben', 'Suffolk', 'Sullivan', 'Tioga', 'Tompkins', 'Ulster', 'Warren', 'Washington', 
                                'Wayne', 'Westchester', 'Westchester', 'Wyoming', 'Yates'))

    elif state == 'NC':
        county = st.sidebar.selectbox('Select your county', ('Alamance', 'Alexander', 'Alleghany', 'Anson', 'Ashe', 'Avery', 'Beaufort', 'Bertie', 
                                'Bladen', 'Brunswick', 'Buncombe', 'Burke', 'Cabarrus', 'Caldwell', 'Camden', 'Carteret', 'Caswell', 
                                'Catawba', 'Chatham', 'Cherokee', 'Chowan', 'Clay', 'Cleveland', 'Columbus', 'Craven', 'Cumberland', 
                                'Currituck', 'Dare', 'Davidson', 'Davie', 'Duplin', 'Durham', 'Edgecombe', 'Forsyth', 'Franklin', 'Gaston', 
                                'Gates', 'Graham', 'Granville', 'Greene', 'Guilford', 'Halifax', 'Harnett', 'Haywood', 'Henderson', 
                                'Hertford', 'Hoke', 'Hyde', 'Iredell', 'Jackson', 'Johnston', 'Jones', 'Lee', 'Lenoir', 'Lincoln', 'Macon', 
                                'Madison', 'Martin', 'McDowell', 'Mecklenburg', 'Mitchell', 'Montgomery', 'Moore', 'Nash', 'New Hanover', 
                                'Northampton', 'Onslow', 'Orange', 'Pamlico', 'Pasquotank', 'Pender', 'Perquimans', 'Person', 'Pitt', 
                                'Polk', 'Randolph', 'Richmond', 'Robeson', 'Rockingham', 'Rowan', 'Rutherford', 'Sampson', 'Scotland', 
                                'Stanly', 'Stokes', 'Surry', 'Swain', 'Transylvania', 'Tyrrell', 'Union', 'Vance', 'Wake', 'Warren', 
                                'Washington', 'Watauga', 'Wayne', 'Wilkes', 'Wilson', 'Yadkin', 'Yancey'))

    elif state == 'ND':
        county = st.sidebar.selectbox('Select your county', ('Adams', 'Barnes', 'Benson', 'Billings', 'Bottineau', 'Bowman', 'Burke', 'Burleigh', 
                                'Cass', 'Cavalier', 'Dickey', 'Divide', 'Dunn', 'Eddy', 'Emmons', 'Foster', 'Golden Valley', 'Grand Forks', 
                                'Grant', 'Griggs', 'Hettinger', 'Kidder', 'LaMoure', 'Logan', 'McHenry', 'McIntosh', 'McKenzie', 'McLean', 
                                'Mercer', 'Morton', 'Mountrail', 'Nelson', 'Oliver', 'Pembina', 'Pierce', 'Ramsey', 'Ransom', 'Renville', 
                                'Richland', 'Rolette', 'Sargent', 'Sheridan', 'Sioux', 'Slope', 'Stark', 'Steele', 'Stutsman', 'Towner', 
                                'Traill', 'Walsh', 'Ward', 'Wells', 'Williams'))

    elif state == 'OH':
        county = st.sidebar.selectbox('Select your county', ('Adams', 'Allen', 'Ashland', 'Ashtabula', 'Athens', 'Auglaize', 'Belmont', 'Brown', 
                                'Butler', 'Carroll', 'Champaign', 'Clark', 'Clermont', 'Clinton', 'Columbiana', 'Coshocton', 'Crawford', 
                                'Cuyahoga', 'Darke', 'Defiance', 'Delaware', 'Erie', 'Fairfield', 'Fayette', 'Franklin', 'Fulton', 'Gallia', 
                                'Geauga', 'Greene', 'Guernsey', 'Hamilton', 'Hancock', 'Hardin', 'Harrison', 'Henry', 'Highland', 'Hocking', 
                                'Holmes', 'Huron', 'Jackson', 'Jefferson', 'Knox', 'Lake', 'Lawrence', 'Licking', 'Logan', 'Lorain', 
                                'Lucas', 'Madison', 'Mahoning', 'Marion', 'Medina', 'Meigs', 'Mercer', 'Miami', 'Monroe', 'Montgomery', 
                                'Morgan', 'Morrow', 'Muskingum', 'Noble', 'Ottawa', 'Paulding', 'Perry', 'Pickaway', 'Pike', 'Portage', 
                                'Preble', 'Putnam', 'Richland', 'Ross', 'Sandusky', 'Scioto', 'Seneca', 'Shelby', 'Stark', 'Summit', 
                                'Trumbull', 'Tuscarawas', 'Union', 'Van Wert', 'Vinton', 'Warren', 'Washington', 'Wayne', 'Williams', 
                                'Wood', 'Wyandot'))

    elif state == 'OK':
        county = st.sidebar.selectbox('Select your county', ('Adair', 'Alfalfa', 'Atoka', 'Beaver', 'Beckham', 'Blaine', 'Bryan', 'Caddo',
                                'Canadian', 'Carter', 'Cherokee', 'Choctaw', 'Cimarron', 'Cleveland', 'Coal', 'Comanche', 'Cotton',
                                'Craig', 'Creek', 'Custer', 'Delaware', 'Dewey', 'Ellis', 'Garfield', 'Garvin', 'Grady', 'Grant',
                                'Greer', 'Harmon', 'Harper', 'Haskell', 'Hughes', 'Jackson', 'Jefferson', 'Johnston', 'Kay', 'Kingfisher', 
                                'Kiowa', 'Latimer', 'Le Flore', 'Lincoln', 'Logan', 'Love', 'Major', 'Marshall', 'Mayes', 'McClain', 
                                'McCurtain', 'McIntosh', 'Murray', 'Muskogee', 'Noble', 'Nowata', 'Okfuskee', 'Oklahoma', 'Okmulgee', 
                                'Osage', 'Ottawa', 'Pawnee', 'Payne', 'Pittsburg', 'Pontotoc', 'Pottawatomie', 'Pushmataha', 'Roger Mills', 
                                'Rogers', 'Seminole', 'Sequoyah', 'Stephens', 'Texas', 'Tillman', 'Tulsa', 'Wagoner', 'Washington', 
                                'Washita', 'Woods', 'Woodward'))

    elif state == 'OR':
        county = st.sidebar.selectbox('Select your county', ('Baker', 'Benton', 'Clackamas', 'Clatsop', 'Columbia', 'Coos', 'Crook',
                                'Curry', 'Deschutes', 'Douglas', 'Gilliam', 'Grant', 'Harney', 'Hood River', 'Jackson',
                                'Jefferson', 'Josephine', 'Klamath', 'Lake', 'Lane', 'Lincoln', 'Linn', 'Malheur', 'Marion',
                                'Morrow', 'Multnomah', 'Polk', 'Sherman', 'Tillamook', 'Umatilla', 'Union', 'Wallowa', 'Wasco', 
                                'Washington', 'Wheeler', 'Yamhill'))

    elif state == 'PA':
        county = st.sidebar.selectbox('Select your county', ('Adams', 'Allegheny', 'Armstrong', 'Beaver', 'Bedford', 'Berks', 'Blair',
                                'Bradford', 'Bucks', 'Butler', 'Cambria', 'Cameron', 'Carbon', 'Centre', 'Chester', 'Clarion',
                                'Clearfield', 'Clinton', 'Columbia', 'Crawford', 'Cumberland', 'Dauphin', 'Delaware', 'Elk', 'Erie', 
                                'Fayette', 'Forest', 'Franklin', 'Fulton', 'Greene', 'Huntingdon', 'Indiana', 'Jefferson', 'Juniata', 
                                'Lackawanna', 'Lancaster', 'Lawrence', 'Lebanon', 'Lehigh', 'Luzerne', 'Lycoming', 'McKean', 'Mercer',
                                'Mifflin', 'Monroe', 'Montgomery', 'Montour', 'Northampton', 'Northumberland', 'Perry',
                                'Philadelphia', 'Pike', 'Potter', 'Schuylkill', 'Snyder', 'Somerset', 'Sullivan', 'Susquehanna',
                                'Tioga', 'Union', 'Venango', 'Warren', 'Washington', 'Wayne', 'Westmoreland', 'Wyoming', 'York'))

    elif state == 'RI':
        county = st.sidebar.selectbox('Select your county', ('Bristol', 'Kent', 'Newport', 'Providence', 'Washington'))

    elif state == 'SC':
        county = st.sidebar.selectbox('Select your county', ('Abbeville', 'Aiken', 'Allendale', 'Anderson', 'Bamberg',
                                'Barnwell', 'Beaufort', 'Beaufort', 'Berkeley', 'Calhoun', 'Charleston', 
                                'Cherokee', 'Chester', 'Chesterfield', 'Clarendon', 'Colleton', 'Darlington', 
                                'Dillon', 'Dorchester', 'Edgefield', 'Fairfield', 'Florence', 'Georgetown',
                                'Greenville', 'Greenwood', 'Hampton', 'Horry', 'Jasper', 'Kershaw', 'Lancaster',
                                'Laurens', 'Lee', 'Lexington', 'Marion', 'Marlboro', 'McCormick', 'Newberry',
                                'Oconee', 'Orangeburg', 'Pickens', 'Richland', 'Saluda', 'Spartanburg', 'Sumter',
                                'Union', 'Williamsburg', 'York'))

    elif state == 'SD':
        county = st.sidebar.selectbox('Select your county', ('Aurora', 'Beadle', 'Bennett', 'Bon Homme', 'Brookings', 'Brown',
                                'Brule', 'Buffalo', 'Butte', 'Campbell', 'Charles Mix', 'Clark', 'Clay', 'Codington',
                                'Corson', 'Custer', 'Davison', 'Day', 'Deuel', 'Dewey', 'Douglas', 'Edmunds',
                                'Fall River', 'Faulk', 'Grant', 'Gregory', 'Haakon', 'Hamlin', 'Hand', 'Hanson',
                                'Harding', 'Hughes', 'Hutchinson', 'Hyde', 'Jackson', 'Jerauld', 'Jones', 'Kingsbury',
                                'Lake', 'Lawrence', 'Lincoln', 'Lyman', 'Marshall', 'McCook', 'McPherson', 'Meade',
                                'Mellette', 'Miner', 'Minnehaha', 'Moody', 'Oglala Lakota', 'Pennington', 'Perkins',
                                'Potter', 'Roberts', 'Sanborn', 'Spink', 'Stanley', 'Sully', 'Todd', 'Tripp', 'Turner',
                                'Union', 'Walworth', 'Yankton', 'Ziebach'))

    elif state == 'TN':
        county = st.sidebar.selectbox('Select your county', ('Anderson', 'Bedford', 'Benton', 'Bledsoe', 'Blount', 'Bradley', 
                                'Campbell', 'Cannon', 'Carroll', 'Carter', 'Cheatham', 'Chester', 'Claiborne', 'Clay',
                                'Cocke', 'Coffee', 'Crockett', 'Cumberland', 'Davidson', 'De Kalb', 'Decatur', 'Dickson',
                                'Dyer', 'Fayette', 'Fentress', 'Franklin', 'Gibson', 'Giles', 'Grainger', 'Greene', 'Grundy',
                                'Hamblen', 'Hamilton', 'Hancock', 'Hardeman', 'Hardin', 'Hawkins', 'Haywood', 'Henderson',
                                'Henry', 'Hickman', 'Houston', 'Humphreys', 'Jackson', 'Jefferson', 'Johnson', 'Knox', 'Lake',
                                'Lauderdale', 'Lawrence', 'Lewis', 'Lincoln', 'Loudon', 'Macon', 'Madison', 'Marion', 'Marshall',
                                'Maury', 'McMinn', 'McNairy', 'Meigs', 'Monroe', 'Montgomery', 'Moore', 'Morgan', 'Obion',
                                'Overton', 'Perry', 'Pickett', 'Polk', 'Putnam', 'Rhea', 'Roane', 'Robertson', 'Rutherford',
                                'Scott', 'Sequatchie', 'Sevier', 'Shelby', 'Smith', 'Stewart', 'Sullivan', 'Sumner', 'Tipton',
                                'Trousdale', 'Unicoi', 'Union', 'Van Buren', 'Warren', 'Washington', 'Wayne', 'Weakley',
                                'White', 'Williamson', 'Wilson'))

    elif state == 'TX':
        county = st.sidebar.selectbox('Select your county', ('Anderson', 'Andrews', 'Angelina', 'Aransas', 'Archer', 'Armstrong',
                                'Atascosa', 'Austin', 'Bailey', 'Bandera', 'Bastrop', 'Baylor', 'Bee', 'Bell', 'Bexar',
                                'Blanco', 'Borden', 'Bosque', 'Bowie', 'Brazoria', 'Brazos', 'Brewster', 'Briscoe',
                                'Brooks', 'Brown', 'Burleson', 'Burnet', 'Caldwell', 'Calhoun', 'Callahan', 'Cameron', 
                                'Camp', 'Carson', 'Cass', 'Castro', 'Chambers', 'Cherokee', 'Childress', 'Clay', 'Cochran', 
                                'Coke', 'Coleman', 'Collin', 'Collingsworth', 'Colorado', 'Comal', 'Comanche', 'Concho', 
                                'Cooke', 'Coryell', 'Cottle', 'Crane', 'Crockett', 'Crosby', 'Culberson', 'Dallam', 'Dallas', 
                                'Dawson', 'DeWitt', 'Deaf Smith', 'Delta', 'Denton', 'Dickens', 'Dimmit', 'Donley', 'Duval', 
                                'Eastland', 'Ector', 'Edwards', 'El Paso', 'Ellis', 'Erath', 'Falls', 'Fannin', 'Fayette',
                                'Fisher', 'Floyd', 'Foard', 'Fort Bend', 'Franklin', 'Freestone', 'Frio', 'Gaines', 'Galveston',
                                'Garza', 'Gillespie', 'Glasscock', 'Goliad', 'Gonzales', 'Gray', 'Grayson', 'Gregg', 'Grimes',
                                'Guadalupe', 'Hale', 'Hall', 'Hamilton', 'Hansford', 'Hardeman', 'Hardin', 'Harris', 'Harrison',
                                'Hartley', 'Haskell', 'Hays', 'Hemphill', 'Henderson', 'Hidalgo', 'Hill', 'Hockley', 'Hood',
                                'Hopkins', 'Houston', 'Howard', 'Hudspeth', 'Hunt', 'Hutchinson', 'Irion', 'Jack', 'Jackson',
                                'Jasper', 'Jeff Davis', 'Jefferson', 'Jim Hogg', 'Jim Wells', 'Johnson', 'Jones', 'Karnes',
                                'Kaufman', 'Kendall', 'Kenedy', 'Kent', 'Kerr', 'Kimble', 'King', 'Kinney', 'Kleberg', 'Knox',
                                'La Salle', 'Lamar', 'Lamb', 'Lampasas', 'Lavaca', 'Lee', 'Leon', 'Liberty', 'Limestone', 'Lipscomb', 
                                'Live Oak', 'Llano', 'Loving', 'Lubbock', 'Lynn', 'Madison', 'Marion', 'Martin', 'Mason', 'Matagorda', 
                                'Maverick', 'McCulloch', 'McLennan', 'McMullen', 'Medina', 'Menard', 'Midland', 'Milam', 'Mills',
                                'Mitchell', 'Montague', 'Montgomery', 'Moore', 'Morris', 'Motley', 'Nacogdoches', 'Navarro', 'Newton',
                                'Nolan', 'Nueces', 'Ochiltree', 'Oldham', 'Orange', 'Palo Pinto', 'Panola', 'Parker', 'Parmer',
                                'Pecos', 'Polk', 'Potter', 'Presidio', 'Rains', 'Randall', 'Reagan', 'Real', 'Red River', 'Reeves', 
                                'Refugio', 'Roberts', 'Robertson', 'Rockwall', 'Runnels', 'Rusk', 'Sabine', 'San Augustine',
                                'San Jacinto', 'San Patricio', 'San Saba', 'Schleicher', 'Scurry', 'Shackelford', 'Shelby', 'Sherman', 
                                'Smith', 'Somervell', 'Starr', 'Stephens', 'Sterling', 'Stonewall', 'Sutton', 'Swisher', 'Tarrant',
                                'Taylor', 'Terrell', 'Terry', 'Throckmorton', 'Titus', 'Tom Green', 'Travis', 'Trinity', 'Tyler',
                                'Upshur', 'Upton', 'Uvalde', 'Val Verde', 'Van Zandt', 'Victoria', 'Walker', 'Waller', 'Ward',
                                'Washington', 'Webb', 'Wharton', 'Wheeler', 'Wichita', 'Wilbarger', 'Willacy', 'Williamson', 'Wilson', 
                                'Winkler', 'Wise', 'Wood', 'Yoakum', 'Young', 'Zapata', 'Zavala'))

    elif state == 'UT':
        county = st.sidebar.selectbox('Select your county', ('Beaver', 'Box Elder', 'Cache', 'Carbon', 'Daggett',
                                'Davis', 'Duchesne', 'Emery', 'Garfield', 'Grand', 'Iron', 'Juab',
                                'Kane', 'Millard', 'Morgan', 'Piute', 'Rich', 'Salt Lake', 'San Juan',
                                'Sanpete', 'Sevier', 'Summit', 'Tooele', 'Uintah', 'Utah', 'Wasatch',
                                'Washington', 'Wayne', 'Weber'))

    elif state == 'VT':
        county = st.sidebar.selectbox('Select your county', ('Addison', 'Bennington', 'Caledonia', 'Chittenden', 
                                'Essex', 'Franklin', 'Grand Isle', 'Lamoille', 'Orange', 'Orleans', 'Rutland',
                                'Washington', 'Windham', 'Windsor'))

    elif state == 'VA':
        county = st.sidebar.selectbox('Select your county', ('Accomack', 'Albemarle', 'Alleghany', 'Amelia', 'Amherst', 
                                'Appomattox', 'Arlington', 'Augusta', 'Bath', 'Bedford', 'Bland', 'Botetourt',
                                'Brunswick', 'Buchanan', 'Buckingham', 'Campbell', 'Caroline', 'Carroll',
                                'Charles City', 'Charlotte', 'Chesterfield', 'Clarke', 'Craig', 'Culpeper',
                                'Cumberland', 'Dickenson', 'Dinwiddie', 'Essex', 'Fairfax', 'Fauquier', 'Floyd',
                                'Fluvanna', 'Franklin', 'Frederick', 'Giles', 'Gloucester', 'Goochland', 'Grayson',
                                'Greene', 'Greensville', 'Halifax', 'Hanover', 'Henrico', 'Henry', 'Highland',
                                'Isle of Wight', 'James City', 'King George', 'King William', 'King and Queen',
                                'Lancaster', 'Lee', 'Loudoun', 'Louisa', 'Lunenburg', 'Madison', 'Mathews',
                                'Mecklenburg', 'Middlesex', 'Montgomery', 'Nelson', 'New Kent', 'Northampton',
                                'Northumberland', 'Nottoway', 'Orange', 'Page', 'Patrick', 'Pittsylvania',
                                'Powhatan', 'Prince Edward', 'Prince George', 'Prince William', 'Pulaski',
                                'Rappahannock', 'Richmond', 'Roanoke', 'Rockbridge', 'Rockingham', 'Russell',
                                'Scott', 'Shenandoah', 'Smyth', 'Southampton', 'Spotsylvania', 'Stafford', 'Surry',
                                'Sussex', 'Tazewell', 'Warren', 'Washington', 'Westmoreland', 'Wise', 'Wythe', 'York'))

    elif state == 'WA':
        county = st.sidebar.selectbox('Select your county', ('Adams', 'Asotin', 'Benton', 'Chelan', 'Clallam',
                                'Clark', 'Columbia', 'Cowlitz', 'Douglas', 'Ferry', 'Franklin', 'Garfield',
                                'Grant', 'Grays Harbor', 'Island', 'Jefferson', 'King', 'Kitsap',
                                'Kittitas', 'Klickitat', 'Lewis', 'Lincoln', 'Mason', 'Okanogan', 'Pacific',
                                'Pend Oreille', 'Pierce', 'San Juan', 'Skagit', 'Skamania', 'Snohomish',
                                'Spokane', 'Stevens', 'Thurston', 'Wahkiakum', 'Walla Walla', 'Whatcom',
                                'Whitman', 'Yakima'
        ))

    elif state == 'WV':
        county = st.sidebar.selectbox('Select your county', ('Barbour', 'Berkeley', 'Boone', 'Braxton', 'Brooke',
                                'Cabell', 'Calhoun', 'Clay', 'Doddridge', 'Fayette', 'Gilmer', 'Grant',
                                'Greenbrier', 'Hampshire', 'Hancock', 'Hardy', 'Harrison', 'Jackson',
                                'Jefferson', 'Kanawha', 'Lewis', 'Lincoln', 'Logan', 'Marion', 'Marshall',
                                'Mason', 'McDowell', 'Mercer', 'Mineral', 'Mingo', 'Monongalia', 'Monroe',
                                'Morgan', 'Nicholas', 'Ohio', 'Pendleton', 'Pleasants', 'Pocahontas',
                                'Preston', 'Putnam', 'Raleigh', 'Randolph', 'Ritchie', 'Roane', 'Summers',
                                'Taylor', 'Tucker', 'Tyler', 'Upshur', 'Wayne', 'Webster', 'Wetzel', 'Wirt',
                                'Wood', 'Wyoming'))

    elif state == 'WI':
        county = st.sidebar.selectbox('Select your county', ('Adams', 'Ashland', 'Barron', 'Bayfield', 'Brown',
                                'Buffalo', 'Burnett', 'Calumet', 'Chippewa', 'Clark', 'Columbia', 'Crawford', 
                                'Dane', 'Dodge', 'Door', 'Douglas', 'Dunn', 'Eau Claire', 'Florence', 
                                'Fond du Lac', 'Forest', 'Grant', 'Green', 'Green Lake', 'Iowa', 'Iron',
                                'Jackson', 'Jefferson', 'Juneau', 'Kenosha', 'Kewaunee', 'La Crosse',
                                'Lafayette', 'Langlade', 'Lincoln', 'Manitowoc', 'Marathon', 'Marinette',
                                'Marquette', 'Menominee', 'Milwaukee', 'Monroe', 'Oconto', 'Oneida', 'Outagamie',
                                'Ozaukee', 'Pepin', 'Pierce', 'Polk', 'Portage', 'Price', 'Racine', 'Richland',
                                'Rock', 'Rusk', 'Sauk', 'Sawyer', 'Shawano', 'Sheboygan', 'St. Croix', 'Taylor',
                                'Trempealeau', 'Vernon', 'Vilas', 'Walworth', 'Washburn', 'Washington', 'Waukesha',
                                'Waupaca', 'Waushara', 'Winnebago', 'Wood'))

    elif state == 'WY':
        county = st.sidebar.selectbox('Select your county', ('Albany', 'Big Horn', 'Campbell', 'Carbon', 'Converse', 'Crook',
                                'Fremont', 'Goshen', 'Hot Springs', 'Johnson', 'Laramie', 'Lincoln', 'Natrona', 'Niobrara',
                                'Park', 'Platte', 'Sheridan', 'Sublette', 'Sweetwater', 'Teton', 'Uinta', 'Washakie',
                                'Weston'))
    
    select_status = st.sidebar.radio("Model type", ('Public Supply Water Withdrawal vs. Domestic Use',
                                                    'Irrigation Water Withdrawn vs. Wastewater Reclaimed', 
                                                    'Total Water Withdrawal vs. Water Withdrawn for Public Supply', 
                                                    'Population vs. Median Income'))
    
    state_data = df[df['state'] == state]
    fips = int(df[(df['state'] == state) & (df['countyname'] == county)]['fips'])



    st.write(f" #### A brief overview of the data relevant to {county} County, {state}:")
    st.markdown(f"- Total population: {int(df[df['fips'] == fips]['population'])}")
    st.markdown(f"- Public supply total withdrawals: {int(df[df['fips'] == fips]['ps_wtotl'])} million gallons per day")
    st.markdown(f"- Domestic deliveries from public supply: {int(df[df['fips'] == fips]['do_psdel'])} million gallons per day")
    st.markdown(f"- Total fresh water withdrawals for irrigation: {int(df[df['fips'] == fips]['ir_wfrto'])} million gallons per day")
    st.markdown(f"- Reclaimed wastewater for crop irrigation: {int(df[df['fips'] == fips]['ir_recww'])} million gallons per day")
    st.markdown(f"- Total withdrawals: {int(df[df['fips'] == fips]['to_wtotl'])} million gallons per day")
    st.markdown(f"- Median household income: ${int(df[df['fips'] == fips]['median_household_income'])}")

### Create kmeans clusters model and corresponding visualization for water withdrawn from public supply ###
    if select_status == 'Public Supply Water Withdrawal vs. Domestic Use':
        st.markdown("## Public Supply Water Withdrawal vs. Public Supply Domestic Use")
        st.markdown("##### Here you can see how much your identified cluster uses water in your homes vs. how much is available.")
        df1 = df.filter(items=['ps_wtotl', 'do_psdel', 'state', 'fips'])
    
        # Define X
        X = df1[['ps_wtotl', 'do_psdel']]

        # Scale data
        sc = StandardScaler()
        Z = sc.fit_transform(X)

        km1 = KMeans(n_clusters=4, n_init='auto', random_state=42)
        km1.fit(Z)

        df1['cluster'] = km1.labels_

        centroids = sc.inverse_transform(km1.cluster_centers_)
        centroids = pd.DataFrame(
            centroids,
            columns=['ps_wtotl', 'do_psdel']
        )


        plt.figure(figsize=(6, 4))

        colors = ["red", "green", "purple", "orange"]
        df1['color'] = df1['cluster'].map(lambda p: colors[p])

        # Plot points
        ax = df1.plot(
            kind="scatter",
            x="ps_wtotl",
            y="do_psdel",
            figsize=(10, 8),
            c=df1['color']
        )

        # Plot Centroids
        centroids.plot(
            kind="scatter",
            x="ps_wtotl",
            y="do_psdel",
            marker="*",
            c=colors,
            s=300,
            edgecolor = 'black',
            ax=ax
        )

        # Labels
        plt.title('Water Supply and Use')
        plt.xlabel('Water Amount Withdrawn for Public Supply (Mgal/d)')
        plt.ylabel('Domestic Use From Public Supply (Mgal/d)')

        st.pyplot()

### Create kmeans clusters model and corresponding visualization for irrigation withdrawal and reclaimed wastewater ###
    if select_status == 'Irrigation Water Withdrawn vs. Wastewater Reclaimed':
        st.markdown("## Irrigation Water Amount Withdrawn vs. Wastewater Reclaimed")
        st.markdown("##### The following model can be used to understand the efficiency of water use in agriculture. By " + 
                "comparing the amount of water withdrawn for irrigation to the amount of wastewater reclaimed, " + 
                "policymakers and managers can see how much water is being wasted in the agricultural sector.")
    
        keep = ['ir_wfrto', 'ir_recww', 'ic_wfrto', 'ic_recww', 'ig_wfrto', 'ig_recww']
        df3 = df.filter(items=keep)
    
        # Define X
        X = df3[keep]

        # Scale data
        sc = StandardScaler()
        Z = sc.fit_transform(X)

        km3 = KMeans(n_clusters=4, n_init='auto', random_state=42)
        km3.fit(Z)

        df3['cluster'] = km3.labels_

        centroids = sc.inverse_transform(km3.cluster_centers_)
        centroids = pd.DataFrame(
            centroids,
            columns=keep
        )

        fig, ax = plt.subplots(1,2, figsize=(16, 6))

        colors = ["red", "green", 'purple', 'orange']
        df3['color'] = df3['cluster'].map(lambda p: colors[p])

        # Plot points
        df3.plot(
            kind="scatter",
            x="ic_wfrto",
            y="ig_wfrto",
            figsize=(10, 8),
            c=df3['color'],
            ax=ax[0]
        )

        # Plot Centroids
        centroids.plot(
            kind="scatter",
            x="ic_wfrto",
            y="ig_wfrto",
            marker="*",
            c=colors,
            s=300,
            edgecolor = 'black',
            ax=ax[0]
        )

        # Labels
        ax[0].set_title('Irrigation Water Withdrawl: Crops vs. Golf')
        ax[0].set_xlabel('Irrigation-Crop Water Amount Withdrawn (Mgal/d)')
        ax[0].set_ylabel('Irrigation-Golf Water Amount Withdrawn (Mgal/d)')



        # Plot points
        df3.plot(
            kind="scatter",
            x="ic_wfrto",
            y="ic_recww",
            figsize=(10, 8),
            c=df3['color'],
            ax=ax[1]
        )

        # Plot Centroids
        centroids.plot(
            kind="scatter",
            x="ic_wfrto",
            y="ic_recww",
            marker="*",
            c=colors,
            s=300,
            edgecolor = 'black',
            ax=ax[1]
        )

        # Labels
        ax[1].set_title('Irrigation Water Amount Reclaimed')
        ax[1].set_xlabel('Irrigation Water Amount Withdrawn (Mgal/d)')
        ax[1].set_ylabel('Irrigation Wastewater Amount Reclaimed (Mgal/d)')

        st.pyplot()

        # if df[df['fips']==fips]['cluster_1'] == 
        # st.write(" #### {county} County's cluster is colored in {color}.")

### Create kmeans clusters model and corresponding visualization for total and public supply withdrawal ###
    if select_status == 'Total Water Withdrawal vs. Water Withdrawn for Public Supply':
        st.markdown("## Total Water Withdrawal vs. Water Withdrawn for Public Supply")
        st.markdown("##### The model below can be used to understand the overall demand for water in a region. By " +
                    "comparing the total amount of water withdrawn to the amount of water withdrawn " +
                    "for public supply, policymakers and managers can see how much water is being used " +
                    "by households, businesses, and industries.")
    
        keep = ['to_wtotl', 'do_psdel', 'ps_wtotl']
        df5 = df.filter(items=keep)
    
        # Define X
        X = df5[keep]

        # Scale data
        sc = StandardScaler()
        Z = sc.fit_transform(X)

        km5 = KMeans(n_clusters=4, n_init='auto', random_state=42)
        km5.fit(Z)

        df5['cluster'] = km5.labels_

        centroids = sc.inverse_transform(km5.cluster_centers_)
        centroids = pd.DataFrame(
            centroids,
            columns=keep
        )

        fig, ax = plt.subplots(1,2)
        fg = (16,8)

        colors = ["red", "green", "purple", "orange"]
        df5['color'] = df5['cluster'].map(lambda p: colors[p])

        # Plot points
        df5.plot(
            kind="scatter",
            x="to_wtotl",
            y="do_psdel",
            figsize=fg,
            c=df5['color'],
            ax=ax[0]
        )
        # Plot Centroids
        centroids.plot(
            kind="scatter",
            x="to_wtotl",
            y="do_psdel",
            marker="*",
            c=colors,
            s=300,
            edgecolor = 'black',
            ax=ax[0]
        )
        # Labels
        ax[0].set_title("Total Water Withdrawal and Domestic Use from Public Supply Delivery", fontsize=13);
        ax[0].set_xlabel("Total Water Withdrawal (Mgal/d)", fontsize=13)
        ax[0].set_ylabel("Domestic Use From Public Supply (Mgal/d)", fontsize=13);



        # Plot points
        df5.plot(
            kind="scatter",
            x="to_wtotl",
            y="ps_wtotl",
            figsize=fg,
            c=df5['color'],
            ax=ax[1]
        )
        # Plot Centroids
        centroids.plot(
            kind="scatter",
            x="to_wtotl",
            y="ps_wtotl",
            marker="*",
            c=colors,
            s=300,
            edgecolor = 'black',
            ax=ax[1]
        )
        # Labels
        ax[1].set_title("Total Water Withdrawal and Public Supply Water Withdrawal", fontsize=13)
        ax[1].set_xlabel("Total Water Withdrawal (Mgal/d)", fontsize=13)
        ax[1].set_ylabel("Public Supply Water Withdrawal", fontsize=13);

        st.pyplot()

### Create kmeans clusters model and corresponding visualization for median household income ###
    if select_status == 'Population vs. Median Income':
        st.markdown("## Population vs. Median Income")
        st.markdown("##### Here you can see what cluster they are in for baseline understanding of socioeconomic " +
                    "considerations, water demand, and resource management.")
    
        keep = ['population', 'median_household_income']
        df6 = df.filter(items=keep)
    
        # Define X
        X = df6[keep]

        # Scale data
        sc = StandardScaler()
        Z = sc.fit_transform(X)

        km6 = KMeans(n_clusters=4, n_init='auto', random_state=42)
        km6.fit(Z)

        df6['cluster'] = km6.labels_

        centroids = sc.inverse_transform(km6.cluster_centers_)
        centroids = pd.DataFrame(
            centroids,
            columns=keep
        )

        colors = ["red", "green", "purple", "orange"]
        df6['color'] = df6['cluster'].map(lambda p: colors[p])

        # Plot points
        ax = df6.plot(
            kind="scatter",
            x="population",
            y="median_household_income",
            figsize=(16,8),
            c=df6['color']
        )

        # Plot Centroids
        fig = centroids.plot(
            kind="scatter",
            x="population",
            y="median_household_income",
            marker="*",
            c=colors,
            s=300,
            edgecolor = 'black',
            ax=ax
        )

        # Labels
        plt.title("Population and Income", fontsize=13);
        plt.xlabel("Population", fontsize=13)
        plt.ylabel("Median Household Income", fontsize=13);

        st.pyplot()


elif page == 'Data Frame':

    df = pd.read_csv('../../data/clean-data/combined.csv')
    dict = pd.read_csv('../../data/clean-data/data_dict.csv')

    datatable = df.sort_values(by='fips', ascending=True)
    st.markdown("Water Usage, Temperature, Drought, and Income Data")
    st.dataframe(datatable)
    st.markdown("Data Dictionary")
    st.dataframe(dict)
