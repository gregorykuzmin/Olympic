import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import altair as alt
import plotly_express as px


with st.echo(code_location='below'):
    @st.cache(allow_output_mutation=True)
    def get_data():
        df = pd.read_csv(r'athlete_events.csv').drop(columns='Medal')
        codes =  pd.read_csv('codes.csv').drop(columns='ISO code')
        return df, codes
    
    def country(code):
        try:
            return codes.loc[codes['NOC'] == code, 'Country'].values[0]
        except Exception:
            return np.nan

    '''
    # Данные о спортсменах Олимпийские игр с 1896 до 2016 года
    ### Посмотрим на структуру данных
    '''

    df, codes = get_data()
    
    st.write(df.head())

    '''
    ### Теперь посмотрим на количество спортсменов в разных странах за выбранный период по выбранным видам спорта
    '''

    min_year = st.slider('Минимальный год', 
        min_value=int(df['Year'].min()), 
        max_value=int(df['Year'].max()), 
        step=1, 
        key='map')
    if min_year != 2016:
        max_year = st.slider('Максимальный год', 
        min_value=int(df['Year'].max()), 
        max_value=min_year,
        step=-1, 
        key='map')
    else:
        max_year = 2016

    sports = st.multiselect('Виды спорта', 
        ['ALL'] + sorted(df[(df['Year'] >= min_year) & (df['Year'] <= max_year)]['Sport'].unique()), 
        default='ALL', key='map')
    if 'ALL' in sports:
        sports = df[(df['Year'] >= min_year) & (df['Year'] <= max_year)]['Sport'].unique()

    temp_df = df[(df['Year'] >= min_year) & (df['Year'] <= max_year)]
    temp_df = temp_df[temp_df['Sport'].isin(sports)]
    temp_df['Количество спортсменов'] = 1
    temp_df = temp_df.groupby('NOC').agg('count').reset_index().sort_values(by=['NOC'], axis=0)
    temp_df['Country'] = temp_df['NOC'].apply(lambda x: country(x))

    if temp_df.shape[0] != 0:
        fig = px.choropleth(temp_df, locations='NOC', projection='natural earth', color='Количество спортсменов', \
                            title=f'Количество спортсменов по выбранным видам спорта с начала {min_year} года по {max_year}', \
                            hover_name='Country', template='plotly_dark', width=700, height=500, )
        st.plotly_chart(fig)
    else:
        st.write('#### За данный период не было призеров по выбранным видам спорта')
    
    '''
    ### Посмотрим на соотношение мужчин и женщин в выбрнных видах спорта
    '''

    sport_bar = st.selectbox('Выберите вид спорта', ['ALL'] + sorted(df['Sport'].unique()), key='hist')
    if sport_bar == 'ALL':
        sport_bar = df['Sport'].unique()
    else :
        sport_bar = [sport_bar]
        
    temp_df = df[df['Sport'].isin(sport_bar)].groupby(['Year', 'Sex']).agg(
        'count').reset_index().loc[:, ['Year', 'Sex', 'ID']]
    for year_ in temp_df['Year'].unique():
        if temp_df[temp_df['Year'] == year_]['Sex'].nunique() == 1:
            if temp_df[temp_df['Year']==year_]['Sex'].unique()[0] == 'F':
                d = {'Year': year_, 'Sex': 'M', 'ID': 0}
                temp_df = temp_df.append(d, ignore_index = True)
            if temp_df[temp_df['Year']==year_]['Sex'].unique()[0] == 'M':
                d = {'Year': year_, 'Sex': 'F', 'ID': 0}
                temp_df = temp_df.append(d, ignore_index = True)

    temp_df.sort_values(by=['Year', 'Sex'], inplace=True)
    temp_df.reset_index(inplace = True, drop = True)
    sex_f_m = {'F': 'Female', 'M': 'Male'}
    temp_df['SEX'] = temp_df['Sex'].apply(lambda x: sex_f_m[x])
    for i in temp_df.index:
        if i % 2 == 1:
            temp_df.loc[i, 'Percantage'] = temp_df.loc[i, 'ID'] / (temp_df.loc[i, 'ID'] + temp_df.loc[i - 1, 'ID'])
        elif i % 2 == 0:
            temp_df.loc[i, 'Percantage'] = temp_df.loc[i, 'ID'] / (temp_df.loc[i, 'ID'] + temp_df.loc[i + 1, 'ID'])
    
    fig = px.bar(temp_df, x='SEX', y='Percantage', title='', animation_frame='Year', animation_group='SEX', range_y=[0,1])
    fig.update_yaxes(tickformat=',.0%')

    st.plotly_chart(fig)

    '''
    ##### В соревнованиях по многим видам спорта женщины не принимали уастия до определенного времени. Мы видим, что со временем соотношение женщин и мужчин сравнивается практически во всех видах спорта.
    ### Проанализируем тенденции изменения среднего роста и веса спортсменов
    ##### Мы используем график с двумя вертикальными осями, чтобы мы могли видеть изменение сразу двух несоизмеримых, но при этом связанных параметров  
    '''

    sport_graph = st.selectbox('Выберите вид спорта', ['ALL'] + sorted(df['Sport'].unique()), key='sport_graph')
    if sport_graph == 'ALL':
        sport_graph = df['Sport'].unique()
    else :
        sport_graph = [sport_graph]
    sex_graph =  st.selectbox('Выберите пол', ['ALL'] + ['Female', 'Male'], key='sport_graph')
    if sex_graph == 'ALL':
        sex_graph = ['Female', 'Male']
    else:
        sex_graph = [sex_graph]

    temp_df = df
    temp_df['SEX'] = temp_df['Sex'].apply(lambda x: sex_f_m[x])
    temp_df = df[(df['SEX'].isin(sex_graph)) & (df['Sport'].isin(sport_graph))]
    temp_df = temp_df.groupby('Year').agg('mean').dropna().reset_index().sort_values(by=['Year'])
    temp_df['Year'] = temp_df['Year'].astype('int64')
    
    fig, ax_1 = plt.subplots()
    fig.patch.set_facecolor('#262730')
    ax_1.set_facecolor('#262730')
    
    color_1 = '#868898'
    ax_1.plot(temp_df['Year'].unique(), temp_df['Height'], color=color_1)
    ax_1.set_xlabel('Year', color='w')
    ax_1.set_ylabel('Height', color=color_1)
    ax_1.set_title('Графики зависимости роста и веса от года', color='w')
    ax_1.tick_params(axis='y', labelcolor=color_1)
    ax_1.tick_params(axis='x', colors='white')

    ax_2 = ax_1.twinx()

    color_2 = '#859A8A'
    ax_2.plot(temp_df['Year'].unique(), temp_df['Weight'], color=color_2)
    ax_2.set_ylabel('Weight', color=color_2)
    ax_2.tick_params(axis='y', labelcolor=color_2)
    st.pyplot(fig)

    '''
    ##### Мы можем сделать интересные выводы, например, о том как менялся рост и вес баскетболистов. Также можно сделать интересное наблюдение - футболисты стали выше и относительно легче за время наблюдений.
    ### Посмотрим на средний возраст спортсменов
    ##### На интерактивном графике линиями представлены возраста спортсменов. Ширина полосы - 2 стандартных отклонения.
    '''

    temp_df_std_male = df[df['Sex'] == 'M'].groupby('Year').agg('std').reset_index().loc[:, ['Year', 'Age']].\
    rename(columns={'Age' : 'Age_std'})
    temp_df_mean_male = df[df['Sex'] == 'M'].groupby('Year').agg('mean').reset_index().loc[:, ['Year', 'Age']].\
    rename(columns={'Age' : 'Age_mean'})
    temp_df_male = pd.merge(temp_df_std_male, temp_df_mean_male, on='Year')
    temp_df_male['std+'] = temp_df_male['Age_mean'] + temp_df_male['Age_std']
    temp_df_male['std-'] = temp_df_male['Age_mean'] - temp_df_male['Age_std']
    temp_df_male['Sex'] = 'Male'

    temp_df_std_female = df[df['Sex'] == 'F'].groupby('Year').agg('std').reset_index().loc[:, ['Year', 'Age']].\
    rename(columns={'Age' : 'Age_std'})
    temp_df_mean_female = df[df['Sex'] == 'F'].groupby('Year').agg('mean').reset_index().loc[:, ['Year', 'Age']].\
    rename(columns={'Age' : 'Age_mean'})
    temp_df_female = pd.merge(temp_df_std_female, temp_df_mean_female, on='Year')
    temp_df_female['std+'] = temp_df_female['Age_mean'] + temp_df_female['Age_std']
    temp_df_female['std-'] = temp_df_female['Age_mean'] - temp_df_female['Age_std']
    temp_df_female['Sex'] = 'Female'

    temp_df = pd.concat([temp_df_male, temp_df_female])
    import altair as alt
    
    selection = alt.selection_multi(fields=['Sex'], bind='legend')

    band_chart = alt.Chart(temp_df, title='Средние значения +- стандартное отклонение').mark_area(opacity=1).encode(
    alt.X('Year:Q'),
    alt.Y('std+:Q', title=''),
    alt.Y2('std-:Q', title='Age'),
    alt.Color('Sex:N'),
    opacity=alt.condition(selection, alt.value(0.7), alt.value(0.1))
    ).add_selection(
    selection
    ).properties(
    width=500,
    height=400
    ).interactive()

    line_chart = alt.Chart(temp_df, title='Средние значения возраста').mark_line().encode(
        alt.X('Year:Q', title='Year'),
        alt.Y('Age_mean', title='Age'),
        alt.Color('Sex:N'),
        opacity=alt.condition(selection, alt.value(1), alt.value(0.2))
    ).add_selection(
        selection
    ).properties(
    width=500,
    height=400
    ).interactive()

    chart = alt.vconcat(line_chart, band_chart).configure(
    numberFormat='.0f'
    ).configure_legend(
    titleFontSize=22, 
    labelFontSize=22
    ).configure_axis(
    labelFontSize=18,
    titleFontSize=18
    ).configure_title(fontSize=22)

    st.altair_chart(chart)

    '''
    ##### Можно сделать вывод, что возраст спортсменок в среднем меньше, чем возраст спортсменов
    ### Source code
    '''
