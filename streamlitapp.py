from urllib.parse import urlencode

import altair as alt
import pandas as pd
import streamlit as st
import numpy as np
from annotated_text import annotated_text
from vega_datasets import data
import json
import re
from PyPDF2 import PdfReader
from collections import defaultdict
from rapidfuzz import fuzz, process
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from google import genai
from pydantic import BaseModel

from datetime import datetime, timedelta


st.set_page_config(layout="wide", page_title='Data Science Career Accelerator')

client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])

add_selectbox = st.sidebar.selectbox("Select a page to explore:",
                                     ('Data Science Job Explorer',
                                      'Data Science Job board',
                                      'Data Science Career Elevator'))

filepath_state_fips_dict = './datasets/state_to_fips.txt'
@st.cache_data
def get_state_fips():
    with open(filepath_state_fips_dict) as f:
        text_data = f.read()

    dictionary = json.loads(text_data)
    return dictionary

filepath_jobs_data = "./datasets/consolidated_jobs_skills_extracted.csv"
@st.cache_data
def load_ds_jobs_data():
    ds_data = pd.read_csv(filepath_jobs_data)
    ds_data['Industry'] = ds_data['Industry'].apply(lambda x: 'NA' if x == '-1' else x)
    return ds_data

filepath_col = "./datasets/cost_of_living_loc.csv"
@st.cache_data
def load_col_data():
    col_data = pd.read_csv(filepath_col)
    return col_data

@st.cache_resource
def fetch_sentence_embedding_model():
    
    from sentence_transformers import SentenceTransformer
    
    model = SentenceTransformer("all-MiniLM-L6-v2")

    return model

def embedding_score(model, resume_skills, jd_skills):
    # embed and average
    if not resume_skills or not jd_skills:
        return 0.0
    r_emb = model.encode(resume_skills, convert_to_numpy=True)
    j_emb = model.encode(jd_skills,     convert_to_numpy=True)
    r_mean = r_emb.mean(axis=0)
    j_mean = j_emb.mean(axis=0)
    # cosine
    cos = np.dot(r_mean, j_mean) / (np.linalg.norm(r_mean)*np.linalg.norm(j_mean))
    return float(cos.clip(0,1))*100  # map [-1,1]→[0,100]

def extract_text_from_pdf(path_or_file) -> str:
    """
    Extracts and returns all text from a PDF file (given by file‐path or file‐like object),
    then collapses any runs of whitespace/newlines into single spaces.
    """
    reader = PdfReader(path_or_file)
    raw_text = []
    for page in reader.pages:
        # page.extract_text() returns None if it finds no text on the page
        page_text = page.extract_text() or ""
        raw_text.append(page_text)
    full_text = "\n".join(raw_text)
    
    # clean up whitespace: turn any sequence of whitespace (spaces/newlines/tabs) into one space
    clean = re.sub(r'\s+', ' ', full_text).strip()
    return clean

MAX_DAILY_CALLS = 100

@st.cache_resource
def _get_usage_bucket():
    # count, and when to reset
    return {"count": 0, "reset_at": datetime.now() + timedelta(days=1)}

def check_and_increment():
    bucket = _get_usage_bucket()
    now = datetime.now()
    # reset if past midnight UTC (or you can choose a reset window)
    if now >= bucket["reset_at"]:
        bucket["count"] = 0
        bucket["reset_at"] = now + timedelta(days=1)
    if bucket["count"] >= MAX_DAILY_CALLS:
        return False
    bucket["count"] += 1
    return True

def extract_skills_gemini(text):

    class Skill_extractor(BaseModel):
        technical_skills: list[str]
        soft_skills: list[str]

    response = client.models.generate_content(
        model="gemini-2.0-flash-lite",
        contents=f"""
        
        You are a resume reviewing expert, I want you to extract all technical skill key-words (like- programming languages, database, etc.) and
        soft skill key-words (like- stakeholder presentation, leadership and management, etc.) present within the text in triple backticks below

        ```
        {resume_text}
        ```
        
        """,
        config={
            "response_mime_type": "application/json",
            "response_schema": list[Skill_extractor],
        },
    )

    skills_extract: list[Skill_extractor] = response.parsed

    technical_skills = skills_extract[0].technical_skills
    soft_skills = skills_extract[0].soft_skills

    return technical_skills, soft_skills

def suggest_missing_keywords(resume_skills, jd_skills, threshold=75):
    """
    Returns a list of JD-skills that are not already (well) represented
    in resume_skills, based on a fuzzy-match threshold.
    
    resume_skills: list of str from your resume
    jd_skills:     list of str from the job description
    threshold:     int between 0–100; lower = more permissive
    """
    # normalize resume skills once
    resume_norm = [s.lower().strip() for s in resume_skills]
    annotated_input = []
    
    for js in jd_skills:
        js_norm = js.lower().strip()
        # find best fuzzy match in resume
        best = process.extractOne(js_norm, resume_norm, scorer=fuzz.token_set_ratio)
        score = best[1] if best else 0
        if score < threshold:
            annotated_input.append(("  " + js.capitalize() + "  ", "", "#faa"))
        else:
            annotated_input.append(("  " + js.capitalize() + "  ", "", "#afa"))

    # for rs in resume_skills:
    #     annotated_input.append(("  " + rs.capitalize() + "  ", "", "#afa"))

    return annotated_input

####################################################################################################################################################################################################
##### DASHBOARD 1: JOB EXPLORER
####################################################################################################################################################################################################

if add_selectbox == 'Data Science Job Explorer':

    state_fips = get_state_fips()
    
    st.title('US Data Science Jobs Explorer')

    # Number of Jobs in United States:
    with st.spinner(text="Loading data..."):
        jobs = load_ds_jobs_data()

    total_jobs = jobs.shape[0]
    st.markdown(f'#### Welcome to the Jobs Explorer, Analyze insights from around *{total_jobs}* Data Science roles in our job board')
    st.markdown(f'Data science sits at the intersection of statistics, computer science and domain expertise, transforming raw data into actionable insights that drive smarter decisions. From forecasting customer behavior and streamlining operations to powering breakthroughs in healthcare, finance and beyond, data scientists are the architects of today’s data-driven world. With organizations of every size racing to unlock the value buried in their data, demand for skilled practitioners has never been stronger. In 2022, building a career in data science means working on cutting-edge problems, collaborating across industries—and enjoying one of the fastest-growing, most highly-rewarded roles in tech.')
    st.markdown(f'---')

    st.markdown(f'## Where are companies hiring for Data Science roles ?')
    st.markdown("""

    This interactive map shows the total number of data-science openings across every U.S. state. Darker shades indicate higher concentrations of roles, with California, 
    Texas and New York leading the pack. Use the **“Select State”** dropdown to drill into any region—once a state is chosen, the charts below will dynamically 
    update to reveal that state’s breakdown of:

    * Open roles by industry (which sectors are hiring most aggressively)

    * Average and range of salaries by industry

    * City-level salary distributions and role counts

    Together, this view helps you quickly identify both the biggest data-science job markets and the specific industries and cities driving demand in each state.
                    
    """)

    state_agg = jobs.groupby(['State']).agg({'min_salary':'min', 'max_salary':'max', 'Job Title':'count'}).reset_index()
    state_agg.rename(columns={'Job Title':'total_roles'}, inplace=True)
    state_agg['id'] = state_agg['State'].map(state_fips)
    state_agg.dropna(subset=['id'], inplace=True)
    state_agg['id'] = state_agg['id'].astype(int)

    ind_agg = jobs.groupby(['State', 'Industry']).agg({'min_salary':'min', 'max_salary':'max', 'Job Title':'count'}).reset_index()
    ind_agg.rename(columns={'Job Title':'total_roles'}, inplace=True)
    ind_agg['id'] = ind_agg['State'].map(state_fips)
    ind_agg.dropna(subset=['id'], inplace=True)
    ind_agg['id'] = ind_agg['id'].astype(int)

    city_agg = jobs.groupby(['State', 'City']).agg({'min_salary':'min', 'max_salary':'max', 'Job Title':'count'}).reset_index()
    city_agg.rename(columns={'Job Title':'total_roles'}, inplace=True)
    city_agg['id'] = city_agg['State'].map(state_fips)
    city_agg.dropna(subset=['id'], inplace=True)
    city_agg['id'] = city_agg['id'].astype(int)

    # --- dropdown widget ---
    states = ['All'] + sorted(state_agg['State'].unique())
    selected = st.selectbox("Select State:", states)

    if selected != 'All':
        filt_state  = state_agg[state_agg['State']==selected]
        filt_ind    = ind_agg[ind_agg['State']==selected]
        filt_city   = city_agg[city_agg['State']==selected]
    else:
        filt_state, filt_ind, filt_city = state_agg, ind_agg, city_agg

    us_states = alt.topo_feature(data.us_10m.url, 'states')

    base = (
        alt.Chart(us_states)
        .mark_geoshape(fill='lightgray', stroke='white')
    )

    # fill based on total_roles, lookup from filtered state_agg
    us_choropleth = alt.Chart(us_states).mark_geoshape().encode(
        color=alt.condition(
            "datum.total_roles!=null",
            'total_roles:Q',
            alt.value('lightgray'),
            scale=alt.Scale(scheme='blues'),
            title='Total Roles'
        ),
        tooltip=[
            alt.Tooltip('State:N'),
            alt.Tooltip('min_salary:Q', title='Min Salary', format=','),
            alt.Tooltip('max_salary:Q', title='Max Salary', format=','),
            alt.Tooltip('total_roles:Q', title='Total Roles')
        ]
    ).transform_lookup(
        lookup='id',
        from_=alt.LookupData(filt_state, 'id',
                            ['State','min_salary','max_salary','total_roles'])
    )

    # highlight selected state with an orange stroke
    if selected != 'All':
        highlight = alt.Chart(us_states).mark_geoshape(
            fill=None, stroke='orange', strokeWidth=3
        ).transform_filter(f"datum.id == {int(state_fips[selected])}")
        us_map = alt.layer(base, us_choropleth, highlight)
    else:
        us_map = alt.layer(base, us_choropleth)

    us_layer_chart = us_map.project('albersUsa').properties(
            width='container', height=500,
            title='Jobs by State: Salary & Role Count'
    )

    if selected == 'All':
        # roll up to national industry totals
        df_ind = (
            ind_agg
            .groupby('Industry', as_index=False)
            .agg({
                'total_roles':'sum',
                'min_salary':'min',
                'max_salary':'max'
            })
        )
    else:
        df_ind = filt_ind  # already filtered by state

    bar = (
        alt.Chart(df_ind)
        .transform_window(
            rank='rank(total_roles)',
            sort=[alt.SortField('total_roles', order='descending')]
        )
        .transform_filter('datum.rank <= 10')
        .mark_bar()
        .encode(
            x=alt.X('total_roles:Q', title='Total Roles'),
            y=alt.Y('Industry:N',
                    sort=alt.EncodingSortField('total_roles', order='descending'),
                    title='Industry',
                    axis=alt.Axis(labelFontSize=12, labelOverlap=False, labelLimit=500)),
            color=alt.Color('total_roles:Q',
                            scale=alt.Scale(scheme='tealblues'),
                            title='Total Roles'),
            tooltip=[
                alt.Tooltip('Industry:N'),
                alt.Tooltip('total_roles:Q', title='Total Roles'),
                alt.Tooltip('min_salary:Q',  title='Min Salary', format=','),
                alt.Tooltip('max_salary:Q',  title='Max Salary', format=','),
            ]
        )
        .properties(
            width='container',
            height=400,
            title='Industry Breakdown'
        )
    )


    # — bar‐range city chart, colored by max_salary —
    city = (
        alt.Chart(filt_city)
        .transform_window(
            rank='rank(total_roles)',
            sort=[alt.SortField('total_roles', order='descending')]
        )
        .transform_filter('datum.rank <= 10')
        .mark_bar(size=14)
        .encode(
            y=alt.Y('City:N',
                    sort=alt.EncodingSortField('total_roles', order='descending'),
                    title='City',
                    axis=alt.Axis(labelFontSize=12, labelOverlap=False, labelLimit=500)),
            x=alt.X('min_salary:Q', title='Min Salary'),
            x2=alt.X2('max_salary:Q', title='Max Salary'),
            color=alt.Color('total_roles:Q',
                            scale=alt.Scale(scheme='greens'),
                            title='Total roles'),
            tooltip=[
                alt.Tooltip('City:N'),
                alt.Tooltip('total_roles:Q', title='Total Roles'),
                alt.Tooltip('min_salary:Q', title='Min Salary', format=','),
                alt.Tooltip('max_salary:Q', title='Max Salary', format=','),
            ]
        )
        .properties(
            width='container',
            height=400,
            title='Top Cities: Salary Ranges'
        )
    )

    st.altair_chart(us_layer_chart, use_container_width=True)

    st.markdown("""

    This bar chart ranks the leading sectors by number of open Data Science positions (for “All” states). Listings without a specified industry appear at the top as “NA.” 
    Beyond those un-classified roles, Computer Hardware Development, Biotech & Pharmaceuticals and Internet & Web Services dominate, each offering 30–35+ openings. 
    Together, the top ten industries capture the bulk of demand—illustrating that organizations from hardware manufacturers to life-science firms are all racing to hire data talent.

    """)

    st.altair_chart(bar, use_container_width=True)

    st.markdown("""

    Here we spotlight the ten U.S. cities with the most data-science jobs. Each horizontal bar stretches from that city’s reported minimum to maximum salary, 
    and its color intensity reflects total role count. San Francisco leads on both fronts—boasting the widest pay band (~$50K–$350K) and the highest number of listings—while Mountain View, 
    Santa Clara and San Jose follow closely, underscoring the Bay Area’s premium compensation and deep hiring pipelines.

    """)

    st.altair_chart(city, use_container_width=True)
    # st.altair_chart(alt.hconcat(bar, city)
    #                 .resolve_legend(color='independent')
    #                 .resolve_scale(color='independent'), use_container_width=True)
    

    st.markdown(f'---')
    st.markdown(f'## How does your current salary compare across various states/cities ?')

    st.write('Select your Current City, State and salary to view ')
    
    col_data = load_col_data()

    col_states_agg = (
        col_data
        .groupby('State', as_index=False)
        .agg({'Cost of Living Index':'mean'})
        .rename(columns={'Cost of Living Index':'avg_col_index'})
    )
    col_states_agg['id'] = col_states_agg['State'].map(state_fips).astype(int)
    # st.write(col_states_agg.head())

    # build the choropleth
    col_state_map = (
        alt.Chart(us_states)                                # us_states = alt.topo_feature(...)
        .transform_lookup(
            lookup='id',
            from_=alt.LookupData(
                col_states_agg, 'id',
                ['State','avg_col_index']
            )
        )
        .mark_geoshape(stroke='white')
        .encode(
            color=alt.condition(
                "datum.avg_col_index != null",
                alt.Color(
                    'avg_col_index:Q',
                    scale=alt.Scale(scheme='plasma'),
                    title='Cost of Living Index'
                ),
                alt.value('lightgray')
            ),
            tooltip=[
                alt.Tooltip('State:N'),
                alt.Tooltip('avg_col_index:Q',
                            title='Cost of Living Index',
                            format='.2f')
            ]
        )
        .project('albersUsa')
        .properties(
            width='container',
            height=450,
            title='Average Cost of Living Index by State'
        )
    )

    st.markdown("""
    
    This is a static view of each state’s average Cost-of-Living Index, computed by taking the mean of multiple major-city indices within that state. 
    Warmer hues (reds/yellows) flag states with the highest average expenses (e.g., California, Hawaii, parts of the Northeast), 
    while cooler purples mark more affordable regions. Use this as a broad, state-level baseline before diving into your personalized salary comparisons below.

    """)

    st.altair_chart(col_state_map, use_container_width=True)

    st.markdown("""
    
    #### Try to input your current place of residence and current annual salary to see how it varies at the State and City level after adjusting it for Cost of Living.

    """)

    col1, col2, col3 = st.columns([1,1,1])

    with col1:
        states = col_data['State'].unique().tolist()
        selected_state = st.selectbox("Select State:", states)
    with col2:
        cities = sorted(col_data[col_data.State == selected_state]['City'].unique())
        selected_city = st.selectbox("Select City:", cities)
    with col3:
        curr_salary = st.number_input("Enter Your Current Salary:")

    if curr_salary > 0:
        col_data_stats = col_data.copy()
        
        selected_col_index = col_data.loc[(col_data.State == selected_state) & (col_data.City == selected_city), 'Cost of Living Index']
        col_data_stats['Salary'] = col_data_stats['Cost of Living Index'].apply(lambda x: int((curr_salary*x)/selected_col_index))


        state_agg_col = col_data_stats.groupby('State').agg({'Salary':['mean', 'min', 'max']}).reset_index()
        state_agg_col['id'] = state_agg_col['State'].map(state_fips).astype(int)
        state_agg_col.columns = ['State', 'mean_salary', 'min_salary', 'max_salary', 'id']
        state_agg_col[['mean_salary', 'min_salary', 'max_salary']] = state_agg_col[['mean_salary', 'min_salary', 'max_salary']].astype(int)
        # st.write(state_agg_col)

        select_state = alt.selection_single(
            fields=['id'],        # bind to the topojson's 'id' field
            empty='all',          # when nothing clicked, pass all through
            on='click'            # click to select
        )

        # --- dropdown widget ---
        states_list = ['All'] + sorted(state_agg_col['State'].unique())
        selected_state_col = st.selectbox("Select State:", states_list)

        if selected_state_col == 'All':
            # choropleth
            state_map = (
                alt.Chart(us_states)
                # 1) Lookup mean_salary into each shape
                .transform_lookup(
                    lookup='id',
                    from_=alt.LookupData(
                        state_agg_col,   # your DataFrame with columns ['id','State','mean_salary']
                        key='id',
                        fields=['State','mean_salary']
                    )
                )
                # 2) Draw the shapes (no constant fill here)
                .mark_geoshape(stroke='white')
                # 3) Encode color by the looked-up mean_salary
                .encode(
                    color=alt.condition(
                        "datum.mean_salary != null",
                        alt.Color('mean_salary:Q',
                                scale=alt.Scale(scheme='blues'),
                                title='Adjusted Salary'),
                        alt.value('lightgray')
                    ),
                    tooltip=[
                        alt.Tooltip('State:N'),
                        alt.Tooltip('mean_salary:Q',
                                    title='Adjusted Salary',
                                    format=',')
                    ]
                )
                .add_selection(select_state)
                .project('albersUsa')
                .properties(
                    width='container',
                    height=500,
                    title='Adjusted Salary by State'
                )
            )

            col_bar = (
                alt.Chart(state_agg_col)
                .transform_window(
                    rank='rank(mean_salary)',
                    sort=[alt.SortField('mean_salary', order='descending')]
                )
                .transform_filter('datum.rank <= 10')
                .mark_bar()
                .encode(
                    x=alt.X('mean_salary:Q', title='Total Roles'),
                    y=alt.Y('State:N',
                            sort=alt.EncodingSortField('mean_salary', order='descending'),
                            title='State',
                            axis=alt.Axis(labelFontSize=12, labelOverlap=False, labelLimit=500)),
                    color=alt.Color('mean_salary:Q',
                                    scale=alt.Scale(scheme='tealblues'),
                                    title='Total Roles'),
                    tooltip=[
                        alt.Tooltip('State:N'),
                        alt.Tooltip('mean_salary:Q', title='Avg Salary'),
                        alt.Tooltip('min_salary:Q',  title='Min Salary', format=','),
                        alt.Tooltip('max_salary:Q',  title='Max Salary', format=','),
                    ]
                )
                .properties(
                    width='container',
                    height=400,
                    title='Industry Breakdown'
                )
            )

        else:

            fips     = int(state_fips[selected_state_col])

            # 2) Prep your city DataFrame (only those in the selected state)
            city_df = col_data_stats[col_data_stats['State'] == selected_state_col]

            city_hover = alt.selection_single(
                fields=['City'],        # match on the City field
                on='mouseover',         # fire on hover
                empty='none'            # when you’re not over a bar, selection is empty
            )

            # 4) Outline the selected state only
            state_outline = (
                alt.Chart(us_states)
                .transform_filter(f"datum.id == {fips}")
                .mark_geoshape(
                    fill='lightseagreen',
                    stroke='black'
                )
            )

            # 5) Bubble layer for that same state
            bubbles = (
                alt.Chart(city_df)
                .mark_circle(opacity=0.7, stroke='white')
                .encode(
                    longitude='lon:Q',
                    latitude='lat:Q',
                    size=alt.Size(
                        'Salary:Q',
                        scale=alt.Scale(range=[100,1500]),
                        title='Adj. Salary'
                    ),
                    color=alt.value('orange'),
                    tooltip=[
                        alt.Tooltip('City:N'),
                        alt.Tooltip('Salary:Q', title='Adj. Salary', format=',')
                    ]
                )
            )

            # 6) Layer them & project together
            state_map = (
                alt.layer(state_outline, bubbles)
                .project('albersUsa')
                .properties(
                    width='container', 
                    height=500,
                    title=f"{selected} — City Salaries"
                )
            )

            col_bar = (
                alt.Chart(city_df)
                .transform_window(
                    rank='rank(Salary)',
                    sort=[alt.SortField('Salary', order='descending')]
                )
                .transform_filter('datum.rank <= 10')
                .mark_bar()
                .encode(
                    x=alt.X('Salary:Q', title='Salary'),
                    y=alt.Y('City:N',
                            sort=alt.EncodingSortField('Salary', order='descending'),
                            title='City',
                            axis=alt.Axis(labelFontSize=12, labelOverlap=False, labelLimit=500)),
                    color=alt.Color('Salary:Q',
                                    scale=alt.Scale(scheme='tealblues'),
                                    title='Salary'),
                    tooltip=[
                        alt.Tooltip('City:N'),
                        alt.Tooltip('Salary:Q', title='Salary', format=',')
                    ]
                )
                .properties(
                    width='container',
                    height=400,
                    title='Industry Breakdown'
                )
            )
        
        st.markdown("""
                    
        When you set “Select State” to **All**, the dashboard shows to a U.S. State-level view of your cost-of-living–adjusted salary, with each state is shaded 
        by how far your current pay would stretch locally (darker = more buying power). The bar chart ranks top 10 states by your adjusted salary, 
        so you can instantly see where your compensation buys the least. Use this national perspective to identify high-value markets before drilling down into 
        any single state’s city-level comparisons.

        When you choose a **specific state**, the map zooms into the major city's within the to show the cost-of-living-adjusted equivalent of your entered salary. 
        Circle size (and color intensity) scales to your “buying-power” pay in that metro, instantly highlighting where your dollars stretch farthest. Below the map, the 
        “City Salary Ranking” bar chart lists the same metros in descending order of adjusted pay—making it easy to compare your compensation potential across every city in your selected state.

        """)

        st.altair_chart(state_map, use_container_width=True)

        st.altair_chart(col_bar, use_container_width=True)

    st.markdown(f'---')
    st.markdown(f'## What Skills are the most saught after in Data Scientists ?')

    st.markdown("""

    This dual word-cloud was generated by running ~1,867 data-science job descriptions through the **Gemini Flash 2.0 Lite LLM**—configured to pull out both 
    technical and functional (soft) skills. On the left, you see core proficiencies like Python, machine learning, SQL and data visualization; on the right, 
    essential interpersonal strengths—communication, leadership, stakeholder presentation and problem-solving. 
    Together, they highlight why **modern data scientists must pair deep technical chops with strong collaboration and strategic thinking to turn raw data into real business impact**.

    """)

    tech_skills_list = [str.lower(skill[1:len(skill)-1]) for jd_skills in jobs.technical_skills.tolist() for skill in jd_skills[1:len(jd_skills)-1].split(', ')]
    
    tech_skills_freq = defaultdict(int)
    for w in tech_skills_list:
        if len(w) > 0:
            tech_skills_freq[w] += 1

    sorted_tech_skills_freq = sorted(tech_skills_freq.items(), key=lambda item: -item[1])
    # sorted_tech_skills_freq[:100] # top 100 skills required

    def group_skills(skills, threshold=80):
        """
        Groups similar skill strings using a fuzzy-match threshold.

        :param skills: List of raw skill strings
        :param threshold: Minimum token_sort_ratio to consider “the same”
        :return: List of sets, each set is a group of similar skills
        """
        ungrouped = set(skills)
        groups = []

        while ungrouped:
            base = ungrouped.pop()
            group = {base}

            # Compare the base skill to all remaining ungrouped skills
            for other in list(ungrouped):
                score = fuzz.token_sort_ratio(base, other)
                if score >= threshold:
                    group.add(other)
                    ungrouped.remove(other)

            groups.append(group)

        return groups

    sorted_tech_skills_list = [i for i,j in sorted_tech_skills_freq[:100]]

    groups = group_skills(sorted_tech_skills_list, threshold=75)

    top_tech_skills_freq = defaultdict(int)
    for group in groups:
        group = list(group)
        group.sort(key=len)
        top_tech_skills_freq[group[0]] = sum([tech_skills_freq[skill] for skill in group])


    # 2. Generate the WordCloud object
    wc_tech_skills = WordCloud(
        width=500,
        height=300,
        background_color='white',
        prefer_horizontal=0.9,
        colormap='PuBu'
    ).generate_from_frequencies(top_tech_skills_freq)

    
    soft_skills_list = [str.lower(skill[1:len(skill)-1]) for jd_skills in jobs.soft_skills.tolist() for skill in jd_skills[1:len(jd_skills)-1].split(', ')]
    
    soft_skills_freq = defaultdict(int)
    for w in soft_skills_list:
        if len(w) > 0:
            soft_skills_freq[w] += 1

    sorted_soft_skills_freq = sorted(soft_skills_freq.items(), key=lambda item: -item[1])
    # sorted_soft_skills_freq[:100] # top 100 skills required

    sorted_soft_skills_list = [i for i,j in sorted_soft_skills_freq[:100]]

    groups = group_skills(sorted_soft_skills_list, threshold=75)

    top_soft_skills_freq = defaultdict(int)
    for group in groups:
        group = list(group)
        group.sort(key=len)
        top_soft_skills_freq[group[0]] = sum([soft_skills_freq[skill] for skill in group])

    # 2. Generate the WordCloud object
    wc_soft_skills = WordCloud(
        width=500,
        height=300,
        background_color='white',
        prefer_horizontal=0.9,
        colormap='Oranges'
    ).generate_from_frequencies(top_soft_skills_freq)

    # Create a figure with 1 row, 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Left subplot: technical skills
    ax1.imshow(wc_tech_skills, interpolation='bilinear')
    ax1.axis('off')
    ax1.set_title("Technical Skills", fontsize=14, pad=10)

    # Right subplot: soft skills
    ax2.imshow(wc_soft_skills, interpolation='bilinear')
    ax2.axis('off')
    ax2.set_title("Soft Skills", fontsize=14, pad=10)

    # Tighten up spacing and render in Streamlit
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown('### Choose the skills you are currently proficient in to know how many jobs you might be eligible for...')

    st.markdown("""

    Use the dropdown above to pick the technical skills you already know— here, “Python” and “Machine Learning.” Behind the scenes, each job description (only 1,491/1867 had meaningful 
    technical skills mentioned in them) was parsed by Gemini Flash 2.0 Lite to extract skills, then checked with RapidFuzz’s 75%-similarity threshold to see if any of 
    your selected proficiencies appear within any posted jobs. The bar chart splits the roles into:

    * **Eligible**: Jobs that list at least one of the skills you selected
    * **Ineligible**: Jobs that do not mention the skills you selected

    This snapshot instantly shows you how many current openings match your skillset—and the above WordClouds can be used as reference to upskill yourself in technical and functional skills 
    to unlock more opportunities.

    """)

    selected_tech_skills = st.multiselect("Technical Skills", top_tech_skills_freq.keys(), default=["python", 'machine learning'],)

    jobs_with_skills = jobs[jobs.technical_skills_cnt > 0].copy()

    def find_eligible(x):

        skills_list = [str.lower(skill[1:len(skill)-1]) for skill in x[1:len(x)-1].split(', ')]
        for skill in selected_tech_skills:
            if any(fuzz.token_sort_ratio(skill, str.lower(cand)) >= 75 for cand in skills_list):
                return 1
        return 0
    
    jobs_with_skills['eligible_flag'] = jobs_with_skills['technical_skills'].apply(find_eligible)

    # st.write(jobs_with_skills[['Job Title','technical_skills', 'eligible_flag']].sample(100))

    total_roles_with_skills = jobs_with_skills.shape[0]
    eligible_roles_with_skills = jobs_with_skills[jobs_with_skills['eligible_flag'] == 1].shape[0]

    chart_df   = pd.DataFrame({
        "Eligibility": ["Eligible", "Ineligible"],
        "Count":       [eligible_roles_with_skills, total_roles_with_skills - eligible_roles_with_skills]
    })

    chart = (
        alt.Chart(chart_df)
        .mark_bar()
        .encode(
            x=alt.X("Eligibility:N", sort=["Eligible","Ineligible"], title=None),
            y=alt.Y("Count:Q", title="Number of Jobs"),
            color=alt.Color("Eligibility:N", legend=None)
        )
        .properties(
            width=400, height=300,
            title=f"Jobs Matching Selected Tech Skills"
        )
    )

    st.write(f"#### Total jobs: **{total_roles_with_skills}**, Eligible: **{eligible_roles_with_skills}**, Ineligible: **{total_roles_with_skills - eligible_roles_with_skills}**")
    st.altair_chart(chart, use_container_width=True)

####################################################################################################################################################################################################
##### DASHBOARD 2: JOB BOARD
####################################################################################################################################################################################################

elif add_selectbox == 'Data Science Job board':

    # Number of Jobs in United States:
    with st.spinner(text="Loading data..."):
        jobs = load_ds_jobs_data()

    total_jobs = jobs.shape[0]

    st.markdown(f"""

    # US Data Science Job board
    
    #### Welcome to the Job board, Explore from around *{total_jobs}* Data Science roles to find your next opportunity
                
    In the Jobs Explorer dashboard, you explored where data-science roles concentrate—by state, city and industry—seen how salaries shift once you factor in cost of living, and 
    uncovered exactly which technical and soft skills recruiters prize today. Now it’s time to bring the focus in, matching your unique résumé and expertise against 
    those insights to surface the opportunities best aligned with your background.

    ---

    """)

    st.markdown('## Data Science Job Board')
    st.markdown("""
    Zero-in on open roles by filtering on State, City, Salary Range and Technical Skills. Below you’ll see how many jobs match your criteria; select any listing to view key details—Company, 
    Rating, Role, Salary Range, Location and Industry—and click “See Job Description” for the full posting.
    """)

    st.markdown('Filter Jobs by:')
    col1, col2, col3 = st.columns([1,1,1])

    with col1:
        states = jobs['State'].unique().tolist()
        selected_state = st.selectbox("Select State:", states, index = states.index('CA'))
    with col2:
        cities = sorted(jobs[jobs.State == selected_state]['City'].unique()) + ['Any']
        selected_city = st.selectbox("Select City:", cities, index = cities.index('Any'))
    with col3:
        est_salary = st.slider("Select expected salary range ($)", jobs.min_salary.min(), jobs.max_salary.max(), (100000.0, 160000.0))


    tech_skills_list = [str.lower(skill[1:len(skill)-1]) for jd_skills in jobs.technical_skills.tolist() for skill in jd_skills[1:len(jd_skills)-1].split(', ')]
    tech_skills_list.remove('')
    filter_tech_skills = st.multiselect("Technical Skills", set(tech_skills_list+['Any']), default=["Any"])

    if selected_city == 'Any':
        jobs_filtered = jobs[(jobs.State == selected_state) & (jobs.min_salary >= est_salary[0]) & (jobs.max_salary <= est_salary[1]) & (jobs.technical_skills_cnt > 0)]
    else:
        jobs_filtered = jobs[(jobs.State == selected_state) & (jobs.City == selected_city) & (jobs.min_salary >= est_salary[0]) & (jobs.max_salary <= est_salary[1]) & (jobs.technical_skills_cnt > 0)]
        
    jobs_filtered.rename(columns={'max_salary':'Max. Salary', 'min_salary':'Min. Salary'}, inplace=True)

    def filter_on_skills(x):

        skills_list = [str.lower(skill[1:len(skill)-1]) for skill in x[1:len(x)-1].split(', ')]
        for skill in skills_list:
            if any(fuzz.token_sort_ratio(skill, str.lower(cand)) >= 75 for cand in filter_tech_skills):
                return 1
        return 0

    if 'Any' in filter_tech_skills:
        jobs_filtered['eligible'] = 1
    else:
        jobs_filtered['eligible'] = jobs_filtered['technical_skills'].apply(filter_on_skills)
    jobs_filtered = jobs_filtered[jobs_filtered['eligible'] == 1]

    jobs_filtered['combined'] = jobs_filtered['Job Title'] + ' at ' + jobs_filtered['Company Name'] + ';    Location: ' + jobs_filtered['Location'] + ';    Salary Range: $' \
                                                                                        + jobs_filtered['Min. Salary'].astype(str) + '- $' + jobs_filtered['Max. Salary'].astype(str)
    
    st.markdown("#### Found {} jobs matching your preferences...".format(jobs_filtered.shape[0]))

    jobs_list = jobs_filtered['combined'].tolist()
    select_job = st.selectbox('Select desired role to view job summary:', jobs_list)

    jobs_selected_index = jobs_list.index(select_job)
    select_job_details = jobs_filtered.iloc[jobs_selected_index, :]

    job_cols = st.columns((1, 1))
    with job_cols[0]:
        st.markdown("**Company Name**: {}".format(select_job_details['Company Name']))
        st.markdown("**Company Rating**: {}/5".format(select_job_details['Company Rating']))
        st.markdown("**Location:** {}".format(select_job_details['Location']))
        st.markdown("**Industry:** {}".format(select_job_details['Industry']))
    with job_cols[1]:
        st.markdown("**Role:** {}".format(select_job_details['Job Title']))
        st.markdown("**Salary:** \${0:,} - \${1:,}".format(int(select_job_details['Min. Salary']), int(select_job_details['Max. Salary'])))
        st.markdown("**Location:** {}".format(select_job_details['Location']))

    with st.expander("See Job Description"):
        st.write(select_job_details['Job Description'])

    st.markdown('---')

    st.markdown('#### ResumeFit: Analyze if your current Resume matches the Job Description of the selected role')

    st.markdown("""

    We use the **Gemini Flash 2.0 Lite LLM** to extract key words from your resume and the selected job description and use a **sentence embedding model** to find cosine similarity 
    between the two sets of keywords to provide the Resume match score. We also suggest key words that can improve the match score by using **fuzzy
    matching** to identify the key-words missing from your resume.

    """)

    uploaded_file = st.file_uploader("Upload your Resume to analyze match with Job Description:")

    # st.write(select_job_details)

    if (uploaded_file is not None) and (select_job_details['technical_skills_cnt'] > 0) and (check_and_increment()):

        resume_text = extract_text_from_pdf(uploaded_file)

        technical_skills, soft_skills = extract_skills_gemini(resume_text)

        skills_resume = technical_skills + soft_skills

        jd_tech_skills_list = [str.lower(skill[1:len(skill)-1])  for skill in select_job_details['technical_skills'][1:len(select_job_details['technical_skills'])-1].split(', ')]
        jd_soft_skills_list = [str.lower(skill[1:len(skill)-1])  for skill in select_job_details['soft_skills'][1:len(select_job_details['soft_skills'])-1].split(', ')]
        skills_jd =  jd_tech_skills_list + jd_soft_skills_list

        sentence_embedding_model = fetch_sentence_embedding_model()

        # st.write(skills_resume)
        # st.write(skills_jd)
        score = embedding_score(sentence_embedding_model, skills_resume, skills_jd)

        if score >= 80:
            indicator = "🟢 Very Good match for the role, you'd be a good fit."
        elif score >= 60:
            indicator = "🟡 Decent Match, can be improved by adding few key words."
        else:
            indicator = "🔴 Not a good match, rework your resume to improve match."

        # display as a single KPI
        st.metric("Your Resume Match score is:", f"{score:.1f}% {indicator}")
        
        suggested_key_words = suggest_missing_keywords(skills_resume, skills_jd, threshold=75)

        annotated_text(*suggested_key_words)

    else:
        st.warning('NOTE: Please upload your resume to perform Resume Matching; Also, ensure there is sufficient job description text available to extract relevant keywords')
        if select_job_details['technical_skills_cnt'] == 0:
            st.error('ERROR: Unable to extract any relevant keywords from the job description. Please try a different role.')
        elif not check_and_increment():
            st.error("🚫 Daily API quota reached. Try again tomorrow.")

####################################################################################################################################################################################################
##### DASHBOARD 1: CAREER ELEVATOR
####################################################################################################################################################################################################

elif add_selectbox == 'Data Science Career Elevator':

    st.markdown("""
                
    # US Data Science Career Elevator
                
    Step into the Career Elevator! —your express ride to finding the perfect role..Upload your resume below, and our AI-powered engine will instantly analyze your skills, experience, and 
    career interests to surface the roles where you’re most likely to excel. In just a few seconds you’ll see:

    * Top-fit job openings tailored to your profile
    * Skill-to-job match scores so you know why each role fits
    * Actionable insights on any gaps and how to close them

    Click “Upload Resume” to get your personalized job recommendations started!
    """)
    
    resume_upload = st.file_uploader("Upload your Resume to start Career Elevator ")

    if (resume_upload is not None)  and (check_and_increment()):

        with st.spinner(text="Starting Career Elevator Recommendation Engine..."):

            resume_text = extract_text_from_pdf(resume_upload)

            technical_skills, soft_skills = extract_skills_gemini(resume_text)

            skills_resume = technical_skills + soft_skills

            sentence_embedding_model = fetch_sentence_embedding_model()

        jobs = load_ds_jobs_data()
        jobs_filtered = jobs[jobs.technical_skills_cnt > 0]
        total_eligible_jobs = jobs_filtered.shape[0]
        
        with st.spinner(text=f"Matching your resume with {total_eligible_jobs} jobs in our database..."):

            def find_bestfit_jobs(row):

                jd_tech_skills_list = [str.lower(skill[1:len(skill)-1])  for skill in row['technical_skills'][1:len(row['technical_skills'])-1].split(', ')]
                jd_soft_skills_list = [str.lower(skill[1:len(skill)-1])  for skill in row['soft_skills'][1:len(row['soft_skills'])-1].split(', ')]
                skills_jd =  jd_tech_skills_list + jd_soft_skills_list

                score = embedding_score(sentence_embedding_model, skills_resume, skills_jd)

                suggested_key_words = suggest_missing_keywords(skills_resume, skills_jd, threshold=75)

                suggested_key_words = [(key_word[0]) for key_word in suggested_key_words if key_word[-1] == '#afa']

                return pd.Series([int(score), suggested_key_words])


            jobs_filtered[['Match Score','Matched Keywords']] = jobs_filtered.apply(find_bestfit_jobs, axis=1)

            jobs_filtered = jobs_filtered[jobs_filtered['Match Score'] >= 70]

            jobs_filtered.rename(columns={'max_salary':'Max Salary', 'min_salary':'Min Salary'}, inplace=True)

            st.markdown('#### Found {} jobs aligned with your profile.'.format(jobs_filtered.shape[0]))

            st.data_editor(
                jobs_filtered[['Job Title', 'Company Name', 'Location', 'Industry', 'Min Salary', 'Max Salary', 'Match Score', 'Matched Keywords']].sort_values(by = 'Match Score', ascending=False),
                column_config={
                    "Matched Keywords": st.column_config.ListColumn(
                    "Matched Keywords",
                    help="Key words from your resume found in the job description",
                    width="medium",
                    ),
                    "Match Score": st.column_config.ProgressColumn(
                        "Match Score",
                        help="Match score between your resume and job description",
                        format=None,
                        min_value=0,
                        max_value=100,
                    )
                },
                hide_index=True,
            )

            scatter = alt.Chart(jobs_filtered).mark_circle(size=60).encode(
                x='Min Salary',
                y='Max Salary',
                color='Match Score',
                tooltip=['Job Title', 'Company Name', 'Location', 'Match Score', 'Min Salary', 'Max Salary']
            ).properties(
                width='container',
                height=400,
                title='Min vs Max Salaries and Match Score'
            ).interactive()

            state_fips = get_state_fips()

            state_agg = jobs_filtered.groupby(['State']).agg({'Min Salary':'min', 'Max Salary':'max', 'Job Title':'count'}).reset_index()
            state_agg.rename(columns={'Job Title':'total_roles'}, inplace=True)
            state_agg['id'] = state_agg['State'].map(state_fips)
            state_agg.dropna(subset=['id'], inplace=True)
            state_agg['id'] = state_agg['id'].astype(int)

            city_agg = jobs_filtered.groupby(['State', 'City']).agg({'Min Salary':'min', 'Max Salary':'max', 'Job Title':'count'}).reset_index()
            city_agg.rename(columns={'Job Title':'total_roles'}, inplace=True)
            city_agg['id'] = city_agg['State'].map(state_fips)
            city_agg.dropna(subset=['id'], inplace=True)
            city_agg['id'] = city_agg['id'].astype(int)

            us_states = alt.topo_feature(data.us_10m.url, 'states')

            base = (
                alt.Chart(us_states)
                .mark_geoshape(fill='lightgray', stroke='white')
            )

            # fill based on total_roles, lookup from filtered state_agg
            us_choropleth = alt.Chart(us_states).mark_geoshape().encode(
                color=alt.condition(
                    "datum.total_roles!=null",
                    'total_roles:Q',
                    alt.value('lightgray'),
                    scale=alt.Scale(scheme='blues'),
                    title='Total Roles'
                ),
                tooltip=[
                    alt.Tooltip('State:N'),
                    alt.Tooltip('Min Salary:Q', title='Min Salary', format=','),
                    alt.Tooltip('Max Salary:Q', title='Max Salary', format=','),
                    alt.Tooltip('total_roles:Q', title='Total Roles')
                ]
            ).transform_lookup(
                lookup='id',
                from_=alt.LookupData(state_agg, 'id',
                                    ['State','Min Salary','Max Salary','total_roles'])
            )

            us_map = alt.layer(base, us_choropleth)
            us_layer_chart = us_map.project('albersUsa').properties(
                    width='container', height=500,
                    title='Jobs by State: Salary & Role Count'
            )

            city = (
                alt.Chart(city_agg)
                .transform_window(
                    rank='rank(total_roles)',
                    sort=[alt.SortField('total_roles', order='descending')]
                )
                .transform_filter('datum.rank <= 20')
                .mark_bar(size=14)
                .encode(
                    y=alt.Y('City:N',
                            sort=alt.EncodingSortField('total_roles', order='descending'),
                            title='City',
                            axis=alt.Axis(labelFontSize=12, labelOverlap=False, labelLimit=500)),
                    x=alt.X('Min Salary:Q', title='Min Salary'),
                    x2=alt.X2('Max Salary:Q', title='Max Salary'),
                    color=alt.Color('total_roles:Q',
                                    scale=alt.Scale(scheme='greens'),
                                    title='Total roles'),
                    tooltip=[
                        alt.Tooltip('City:N'),
                        alt.Tooltip('State:N'),
                        alt.Tooltip('total_roles:Q', title='Total Roles'),
                        alt.Tooltip('Min Salary:Q', title='Min Salary', format=','),
                        alt.Tooltip('Max Salary:Q', title='Max Salary', format=','),
                    ]
                )
                .properties(
                    width='container',
                    height=600,
                    title='Top Cities: Salary Ranges'
                )
            )

            st.markdown("""

            #### Best Match Jobs by Max Salary vs Min Salary
                        
            This scatter plot maps each role’s pay band—minimum salary on the x-axis and maximum on the y-axis—while color intensity reflects how closely it matches your profile. 
            Use it to pinpoint where your best-fit opportunities lie, and to spot high-pay roles with lower match scores so you know which skills to strengthen to reach them.

            """)

            st.altair_chart(scatter, use_container_width=True)

            st.markdown("""
                        
            #### Best Match Jobs by State

            This interactive U.S. map highlights where your top-fit roles are located. 
            * Color intensity shows the total number of matched jobs in each state (darker = more roles). 
            * Hover over a state to see its exact role count and the range of minimum/maximum salaries.
            Use this view to quickly pinpoint geographic hotspots for your best-fit opportunities.

            """)

            st.altair_chart(us_layer_chart, use_container_width=True)


            st.markdown("""

            #### Best Match Jobs by Cities across the US
                        
            This floating-bar chart shows, for each of your top cities:

            * Bar span from Min Salary → Max Salary, so you can compare compensation bands at a glance.
            * Color intensity indicates the number of matched roles in that city (darker = more opportunities).
            * Hover details reveal exact min/max figures and role counts.

            Use this view to spot which cities offer the best pay and the greatest volume of fits.

            """)

            st.altair_chart(city, use_container_width=True)

    else:
        st.warning('NOTE: Please upload your resume to start the Career Elevator job recommendation engine.')
        if not check_and_increment():
            st.error("🚫 Daily API quota reached. Try again tomorrow.")
        
