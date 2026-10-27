import streamlit as st
import pandas as pd
import numpy as np
import requests, time
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import plotly.express as px
import folium
from streamlit_folium import st_folium

st.set_page_config(page_title="Real-World AI Business Location Selector", page_icon="📍", layout="wide")

HEADERS={"User-Agent":"AI-Business-Location-Selector/1.0 student-project"}

@st.cache_data(ttl=86400)
def geocode(q):
    url="https://nominatim.openstreetmap.org/search"
    r=requests.get(url,params={"q":q,"format":"json","limit":1},headers=HEADERS,timeout=20)
    r.raise_for_status()
    a=r.json()
    if not a: return None
    return {"lat":float(a[0]["lat"]),"lon":float(a[0]["lon"]),"display":a[0]["display_name"]}

@st.cache_data(ttl=21600)
def osm_features(lat,lon,business):
    # Real nearby OSM data. Counts are used as measurable location signals.
    radius=1200
    amenity_tags={
        "Café":"cafe|restaurant|fast_food",
        "Restaurant":"restaurant|fast_food|cafe",
        "Grocery Store":"supermarket|convenience|greengrocer",
        "Pharmacy":"pharmacy",
        "Clothing Store":"clothes|fashion|department_store",
        "Salon":"hairdresser|beauty",
        "Electronics":"electronics|computer",
        "Startup / Office":"office|coworking_space"
    }
    tags=amenity_tags.get(business,"shop|amenity")
    query=f"""[out:json][timeout:25];
(
 node(around:{radius},{lat},{lon})[amenity~"^{tags}$"];
 way(around:{radius},{lat},{lon})[amenity~"^{tags}$"];
 node(around:{radius},{lat},{lon})[shop~"^{tags}$"];
 way(around:{radius},{lat},{lon})[shop~"^{tags}$"];
);
out center tags;"""
    r=requests.post("https://overpass-api.de/api/interpreter",data=query,headers=HEADERS,timeout=40)
    r.raise_for_status()
    data=r.json().get("elements",[])
    return len(data)

@st.cache_data(ttl=21600)
def general_osm(lat,lon):
    radius=1500
    query=f"""[out:json][timeout:25];
(
 node(around:{radius},{lat},{lon})[highway~"bus_stop|tram_stop|station"];
 node(around:{radius},{lat},{lon})[amenity~"school|college|university|hospital|clinic|bank|fuel|marketplace"];
 way(around:{radius},{lat},{lon})[amenity~"school|college|university|hospital|clinic|bank|fuel|marketplace"];
);
out center tags;"""
    r=requests.post("https://overpass-api.de/api/interpreter",data=query,headers=HEADERS,timeout=40)
    r.raise_for_status()
    return r.json().get("elements",[])

@st.cache_data(ttl=3600)
def weather(lat,lon):
    url="https://api.open-meteo.com/v1/forecast"
    params={"latitude":lat,"longitude":lon,"current":"temperature_2m,precipitation,wind_speed_10m","timezone":"auto"}
    r=requests.get(url,params=params,timeout=20); r.raise_for_status()
    return r.json().get("current",{})

def minmax(s, reverse=False):
    s=pd.Series(s,dtype=float)
    if s.max()==s.min(): z=pd.Series(50.0,index=s.index)
    else: z=(s-s.min())/(s.max()-s.min())*100
    return 100-z if reverse else z

def make_score(df,w):
    z=df.copy()
    z["Demand Signal"]=(
        .45*z["poi_activity"]+.25*z["population_signal"]+.20*z["transport_count"]+.10*z["facility_count"]
    )
    z["Demand Signal"]=minmax(z["Demand Signal"])
    z["Accessibility"]=minmax(z["transport_count"]+z["road_signal"])
    z["Footfall Proxy"]=minmax(z["poi_activity"]+z["transport_count"])
    z["Competition Opportunity"]=minmax(z["competitor_count"],True)
    z["Rent Affordability"]=minmax(z["rent_affordability"])
    z["Facilities"]=minmax(z["facility_count"])
    z["Purchasing Power"]=z["purchasing_power"]
    z["Location Score"]=sum(w[k]*z[k] for k in w)
    return z.sort_values("Location Score",ascending=False).reset_index(drop=True)

st.title("📍 Real-World AI Business Location Selector")
st.markdown("**Find the strongest candidate area for a shop, restaurant, pharmacy, salon, office or startup using live geographic data.**")

with st.sidebar:
    st.header("1. Business Setup")
    business=st.selectbox("Business type",["Café","Restaurant","Grocery Store","Pharmacy","Clothing Store","Salon","Electronics","Startup / Office"])
    budget=st.number_input("Maximum monthly rent (₹)",min_value=0,value=40000,step=5000)
    st.caption("Enter rent from an actual property listing/landlord quote. The app does not invent rent.")
    st.header("2. Candidate Locations")
    city=st.text_input("City / study area","Bhopal, Madhya Pradesh, India")
    candidates=[]
    for i in range(1,6):
        q=st.text_input(f"Candidate {i}", value="" if i>3 else ["MP Nagar, Bhopal, Madhya Pradesh, India","New Market, Bhopal, Madhya Pradesh, India","Arera Colony, Bhopal, Madhya Pradesh, India"][i-1])
        if q.strip(): candidates.append(q.strip())
    st.header("3. Priorities")
    priority_options=["Customer demand","Accessibility","Low competition","Low rent","Facilities","Purchasing power"]
    selected=st.multiselect("Prioritize",priority_options,default=["Customer demand","Accessibility","Low rent"])

st.info("The application uses live OpenStreetMap/Overpass geographic features and Open-Meteo weather. Population is requested from the geocoding result where available; rent is supplied by the user. Public APIs can have rate limits and incomplete coverage.")

if st.button("🔎 Analyze & Recommend Best Location",type="primary",use_container_width=True):
    if len(candidates)<2:
        st.error("Enter at least 2 candidate locations.")
        st.stop()

    rows=[]; progress=st.progress(0)
    errors=[]
    for idx,q in enumerate(candidates):
        try:
            g=geocode(q)
            if not g: raise ValueError("Location not found")
            comp=osm_features(g["lat"],g["lon"],business)
            elems=general_osm(g["lat"],g["lon"])
            transport=sum(1 for e in elems if e.get("tags",{}).get("highway") in ["bus_stop","station","tram_stop"])
            facilities=sum(1 for e in elems if e.get("tags",{}).get("amenity") in ["school","college","university","hospital","clinic","bank","fuel","marketplace"])
            road=sum(1 for e in elems if "highway" in e.get("tags",{}))
            # Real-world measurable proxy signals; not claimed as observed pedestrian counts.
            poi_activity=comp+facilities
            # Population from geocoder is contextual, not a 1.2 km catchment census.
            pop=g.get("population",0) or 0
            population_signal=np.log1p(pop) if pop else 0
            weather_now=weather(g["lat"],g["lon"])
            rows.append({
                "candidate":q,"resolved_location":g["display"],"latitude":g["lat"],"longitude":g["lon"],
                "competitor_count":comp,"transport_count":transport,"facility_count":facilities,
                "road_signal":road,"poi_activity":poi_activity,"population":pop,
                "population_signal":population_signal,"rent_affordability":max(0,100-(budget/50000*100)),
                "purchasing_power":50.0,"temperature":weather_now.get("temperature_2m",np.nan)
            })
        except Exception as e:
            errors.append(f"{q}: {e}")
        progress.progress((idx+1)/len(candidates))

    if errors:
        st.warning("Some candidates could not be analyzed:\n\n"+"\n".join(errors))
    if len(rows)<2:
        st.error("Not enough successfully resolved locations.")
        st.stop()

    data=pd.DataFrame(rows)
    # User-supplied rent is not a property-level lookup. Ask for actual rent per candidate in a second table.
    st.session_state["locations"]=data
    st.success("Live geographic analysis completed. Enter actual monthly rent for each candidate below, then calculate the final ranking.")

if "locations" in st.session_state:
    data=st.session_state["locations"].copy()
    st.subheader("💰 Enter Real Property Rent")
    rent_default=int(budget or 0)
    edited=st.data_editor(
        data[["candidate","resolved_location","competitor_count","transport_count","facility_count"]].assign(
            monthly_rent=[rent_default]*len(data)
        ),
        column_config={"monthly_rent":st.column_config.NumberColumn("Actual monthly rent (₹)",min_value=0,step=1000)},
        disabled=["candidate","resolved_location","competitor_count","transport_count","facility_count"],
        hide_index=True,use_container_width=True
    )
    data["monthly_rent"]=edited["monthly_rent"].astype(float)
    data["rent_affordability"]=np.clip(100-data.monthly_rent/max(budget,1)*100,0,100) if budget else 50

    # Data-driven default weights, then user priority adjustments.
    w={"Demand Signal":.30,"Accessibility":.18,"Footfall Proxy":.16,"Competition Opportunity":.12,"Rent Affordability":.12,"Facilities":.07,"Purchasing Power":.05}
    if selected:
        boosts={"Customer demand":"Demand Signal","Accessibility":"Accessibility","Low competition":"Competition Opportunity","Low rent":"Rent Affordability","Facilities":"Facilities","Purchasing power":"Purchasing Power"}
        for item in selected: w[boosts[item]]+=.07
        total=sum(w.values()); w={k:v/total for k,v in w.items()}

    result=make_score(data,w)
    result["Rank"]=range(1,len(result)+1)

    st.subheader("🏆 AI Recommendation")
    best=result.iloc[0]
    st.success(f"### Best Location: {best.candidate} — {best['Location Score']:.1f}/100")
    a,b,c,d=st.columns(4)
    a.metric("Suitability",f"{best['Location Score']:.1f}/100")
    b.metric("Competition",int(best.competitor_count))
    c.metric("Transport POIs",int(best.transport_count))
    d.metric("Nearby Facilities",int(best.facility_count))

    st.subheader("🤖 Recommendation Explanation")
    strengths=[]
    if best["Demand Signal"]>=70: strengths.append("strong local activity/demand signal")
    if best["Accessibility"]>=70: strengths.append("good transport and road accessibility")
    if best["Competition Opportunity"]>=70: strengths.append("lower competitor saturation")
    if best["Rent Affordability"]>=70: strengths.append("good rent affordability")
    if best["Facilities"]>=70: strengths.append("many nearby useful facilities")
    st.write("**Why it ranked first:** "+("; ".join(strengths) if strengths else "it has the best overall weighted balance among the candidates.") )
    st.caption("The demand and footfall fields here are proxies derived from observable geographic features; they are not claimed to be actual sales or pedestrian-counter measurements.")

    st.subheader("📊 Candidate Ranking")
    cols=["Rank","candidate","monthly_rent","competitor_count","transport_count","facility_count","Demand Signal","Accessibility","Footfall Proxy","Rent Affordability","Location Score"]
    st.dataframe(result[cols].round(1),hide_index=True,use_container_width=True)

    st.plotly_chart(px.bar(result,x="candidate",y="Location Score",text=result["Location Score"].round(1),title="Real-World Candidate Suitability"),use_container_width=True)
    st.plotly_chart(px.scatter(result,x="monthly_rent",y="Location Score",size="poi_activity",color="competitor_count",hover_name="candidate",title="Rent vs Suitability"),use_container_width=True)

    st.subheader("🗺️ Real Candidate Map")
    m=folium.Map([result.latitude.mean(),result.longitude.mean()],zoom_start=12)
    for _,r in result.iterrows():
        folium.Marker([r.latitude,r.longitude],tooltip=f"{r.candidate}: {r['Location Score']:.1f}/100",
                      popup=f"<b>{r.candidate}</b><br>Score: {r['Location Score']:.1f}/100<br>Competitors: {int(r.competitor_count)}<br>Transport: {int(r.transport_count)}").add_to(m)
    st_folium(m,height=500)

    st.subheader("📥 Export")
    st.download_button("Download Recommendation Report CSV",result.to_csv(index=False).encode(),"real_world_business_location_results.csv","text/csv")

st.divider()
st.caption("Data sources: OpenStreetMap/Overpass for mapped points and transport features; Open-Meteo for current weather; geocoding service for coordinates and contextual population when available. Always verify property rent, footfall, permits, demographics and local commercial conditions before investing.")
