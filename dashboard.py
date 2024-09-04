from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
import requests

def create_dash_app(flask_app, requests_pathname_prefix='/dashboard/'):
    dash_app = Dash(__name__, server=flask_app, url_base_pathname=requests_pathname_prefix,
                    requests_pathname_prefix=requests_pathname_prefix)

    dash_app.layout = html.Div([
        html.H1("Аналитика чат-бота"),
        dcc.Interval(
            id='interval-component',
            interval=60*1000,  # обновление каждую минуту
            n_intervals=0
        ),
        html.Div(id='metrics-container'),
        dcc.Graph(id='queries-over-time'),
        dcc.Graph(id='queries-by-day'),
        dcc.Graph(id='queries-by-week'),
        dcc.Graph(id='queries-by-month'),
        dcc.Graph(id='top-questions'),
        dcc.Graph(id='feedback-distribution')
    ])

    @dash_app.callback(
        [Output('metrics-container', 'children'),
         Output('queries-over-time', 'figure'),
         Output('queries-by-day', 'figure'),
         Output('queries-by-week', 'figure'),
         Output('queries-by-month', 'figure'),
         Output('top-questions', 'figure'),
         Output('feedback-distribution', 'figure')],
        Input('interval-component', 'n_intervals')
    )
    def update_metrics(n):
        try:
            response = requests.get('http://176.109.109.61:8080/analytics_data')
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as e:
            print(f"Ошибка при запросе к API: {e}")
            return html.Div("Ошибка при получении данных"), {}, {}, {}, {}, {}, {}

        metrics = [
            html.P(f"Всего запросов: {data['total_queries']}"),
            html.P(f"Успешных запросов: {data['successful_queries']}"),
            html.P(f"Процент успешных запросов: {data['success_rate']:.2f}%"),
        ]
        
        # График запросов по времени
        queries_df = pd.DataFrame(list(data['queries_over_time'].items()), columns=['time', 'count'])
        queries_df['time'] = pd.to_datetime(queries_df['time'])
        queries_fig = px.line(queries_df, x='time', y='count', title="Запросы по времени")
        
        # График запросов по дням
        queries_by_day = pd.DataFrame(list(data['queries_by_day'].items()), columns=['day', 'count'])
        queries_by_day['day'] = pd.to_datetime(queries_by_day['day'])
        queries_by_day_fig = px.bar(queries_by_day, x='day', y='count', title="Запросы по дням")
        
        # График запросов по неделям
        queries_by_week = pd.DataFrame(list(data['queries_by_week'].items()), columns=['week', 'count'])
        queries_by_week_fig = px.bar(queries_by_week, x='week', y='count', title="Запросы по неделям")
        
        # График запросов по месяцам
        queries_by_month = pd.DataFrame(list(data['queries_by_month'].items()), columns=['month', 'count'])
        queries_by_month_fig = px.bar(queries_by_month, x='month', y='count', title="Запросы по месяцам")
        
        # График топ-10 вопросов
        top_questions = pd.DataFrame(data['top_questions'], columns=['question', 'count'])
        top_questions = top_questions.sort_values('count', ascending=True).tail(10)
        top_questions_fig = px.bar(top_questions, x='count', y='question', orientation='h', title="Топ-10 вопросов")
        
        # График распределения обратной связи
        feedback_df = pd.DataFrame(list(data['feedback_distribution'].items()), columns=['feedback', 'count'])
        feedback_fig = px.pie(feedback_df, values='count', names='feedback', title="Распределение обратной связи")
        
        return metrics, queries_fig, queries_by_day_fig, queries_by_week_fig, queries_by_month_fig, top_questions_fig, feedback_fig

    return dash_app