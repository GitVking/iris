from pytrends.request import TrendReq
import plotly.express as px

pytrends = TrendReq(hl = 'en-US', tz = 360)
pytrends.build_payload(kw_list = ['data science'])

df = pytrends.interest_over_time()
px.line(df['data science'])