---
title: "Feature Engineering with Dates"
tags: [pre-process, features, pandas]
header:
  image: "/images/tutorials/dates.jpg"
  teaser: "/images/teasers/dates.jpg"  
excerpt: "Given a single feature that contains date and time, I created a function that generates 23 features."
mathjax: "true"
categories:
  - Data Processing
---

### Feature Engineering with Dates

In this tutorial I present the *datetime* format that Pandas provides to handle datetime features. In the end I create a function that generates **23** features from a **single** one.

Dates and Times are very precious features for businesses and sales. While a single column can't help us make meaningful insights and take important decisions, creating a handful of features from it can transform data into profit.

### Import Libraries


```python
import numpy as np
import pandas as pd
import datetime

pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)
pd.set_option('max_colwidth', 100)
```


```python
data = pd.read_csv('dataset_with_dates.csv')
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>card_id</th>
      <th>purchase_date</th>
      <th>city_id</th>
      <th>category_1</th>
      <th>category_2</th>
      <th>category_3</th>
      <th>month_lag</th>
      <th>purchase_amount</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>C_ID_415bb3a509</td>
      <td>3/11/2018 14:57</td>
      <td>107</td>
      <td>N</td>
      <td>1.0</td>
      <td>B</td>
      <td>1</td>
      <td>-0.557574</td>
    </tr>
    <tr>
      <th>1</th>
      <td>C_ID_415bb3a509</td>
      <td>3/19/2018 18:53</td>
      <td>140</td>
      <td>N</td>
      <td>1.0</td>
      <td>B</td>
      <td>1</td>
      <td>-0.569580</td>
    </tr>
    <tr>
      <th>2</th>
      <td>C_ID_415bb3a509</td>
      <td>4/26/2018 14:08</td>
      <td>330</td>
      <td>N</td>
      <td>1.0</td>
      <td>B</td>
      <td>2</td>
      <td>-0.551037</td>
    </tr>
    <tr>
      <th>3</th>
      <td>C_ID_415bb3a509</td>
      <td>3/7/2018 9:43</td>
      <td>-1</td>
      <td>Y</td>
      <td>NaN</td>
      <td>B</td>
      <td>1</td>
      <td>-0.671926</td>
    </tr>
    <tr>
      <th>4</th>
      <td>C_ID_ef55cf8d4b</td>
      <td>3/22/2018 21:07</td>
      <td>-1</td>
      <td>Y</td>
      <td>NaN</td>
      <td>B</td>
      <td>1</td>
      <td>-0.659904</td>
    </tr>
  </tbody>
</table>
</div>



### First Step
Convert date to [datetime](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.to_datetime.html) format using Pandas.

After conversion each Series (one column) can be used to access and return several datetime properties like *Series.dt.property


```python
data['purchase_date'] = pd.to_datetime(data['purchase_date']) 
data.head(1)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>card_id</th>
      <th>purchase_date</th>
      <th>city_id</th>
      <th>category_1</th>
      <th>category_2</th>
      <th>category_3</th>
      <th>month_lag</th>
      <th>purchase_amount</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>C_ID_415bb3a509</td>
      <td>2018-03-11 14:57:00</td>
      <td>107</td>
      <td>N</td>
      <td>1.0</td>
      <td>B</td>
      <td>1</td>
      <td>-0.557574</td>
    </tr>
  </tbody>
</table>
</div>



### Properties

**Series.dt.date** - Returns the date part of datetime without timezone information


```python
data['purchase_date'].dt.date.head(1)
```




    0    2018-03-11
    Name: purchase_date, dtype: object



**Series.dt.time** - Returns the time part of datetime


```python
data['purchase_date'].dt.time.head(1)
```




    0    14:57:00
    Name: purchase_date, dtype: object



**Series.dt.year** - Returns the year (int) of the datetime


```python
data['purchase_date'].dt.year.head(1)
```




    0    2018
    Name: purchase_date, dtype: int64



**Series.dt.month** - Returns the month (int) of the datetime


```python
data['purchase_date'].dt.month.head(1)
```




    0    3
    Name: purchase_date, dtype: int64



**Series.dt.day** - Returns the day (int) of the datetime


```python
data['purchase_date'].dt.day.head(1)
```




    0    11
    Name: purchase_date, dtype: int64



**Series.dt.hour** - Returns the hour (int) of the datetime


```python
data['purchase_date'].dt.hour.head(1)
```




    0    14
    Name: purchase_date, dtype: int64



**Series.dt.minute** - Returns the minute (int) of the datetime


```python
data['purchase_date'].dt.minute.head(1)
```




    0    57
    Name: purchase_date, dtype: int64



**Series.dt.second/microsecond/nanosecond** - Returns the second/microsecond/nanosecond (int) of the datetime


```python
data['purchase_date'].dt.second.head(1)
```




    0    0
    Name: purchase_date, dtype: int64



**Series.dt.week/weekofyear** - Returns the week number of the year. Both properties are the same


```python
data['purchase_date'].dt.week.head(1)
```




    0    10
    Name: purchase_date, dtype: int64



**Series.dt.dayofweek/weekday** - Returns the day of the week with Monday=0 and Sunday=6. Both properties are the same


```python
data['purchase_date'].dt.dayofweek.head(1)
```




    0    6
    Name: purchase_date, dtype: int64



**Series.dt.dayofyear** - Returns the ordinal day of the year


```python
data['purchase_date'].dt.dayofyear.head(1)
```




    0    70
    Name: purchase_date, dtype: int64



**Series.dt.quarter** - Returns the quarter of the date


```python
data['purchase_date'].dt.quarter.head(1)
```




    0    1
    Name: purchase_date, dtype: int64



**Series.dt.is_month_start** - Indicator for whether the date is the first day of the month


```python
data['purchase_date'].dt.is_month_start.head(1)
```




    0    False
    Name: purchase_date, dtype: bool



**Series.dt.is_month_end** - Indicator for whether the date is the last day of the month


```python
data['purchase_date'].dt.is_month_end.head(1)
```




    0    False
    Name: purchase_date, dtype: bool



**Series.dt.is_quarter_start** - Indicator for whether the date is the first day of a quarter


```python
data['purchase_date'].dt.is_quarter_start.head(1)
```




    0    False
    Name: purchase_date, dtype: bool



**Series.dt.is_quarter_end** - Indicator for whether the date is the last day of a quarter


```python
data['purchase_date'].dt.is_quarter_end.head(1)
```




    0    False
    Name: purchase_date, dtype: bool



**Series.dt.is_year_start** - Indicate whether the date is the first day of a year


```python
data['purchase_date'].dt.is_year_start.head(1)
```




    0    False
    Name: purchase_date, dtype: bool



**Series.dt.is_year_end** - Indicate whether the date is the last day of a year


```python
data['purchase_date'].dt.is_year_end.head(1)
```




    0    False
    Name: purchase_date, dtype: bool



**Series.dt.is_leap_year** - Indicate whether the date belongs to a leap year


```python
data['purchase_date'].dt.is_leap_year.head(1)
```




    0    False
    Name: purchase_date, dtype: bool



### Feature Engineering


```python
# Takes a dataframe and the column name as inputs
def create_datetime_features(df, c):
    if (df[c].dtype!='datetime64[ns]'):
        pd.to_datetime(df[c]) 
        
    df['year'] = df[c].dt.year
    df['month'] = data[c].dt.month
    df['day'] = data[c].dt.day
    df['hour'] = data[c].dt.hour
    df['session'] = df['hour'].apply(lambda x: get_session(x))
    df['minute'] = data[c].dt.minute
    df['second'] = data[c].dt.second
    df['week'] = data[c].dt.week
    df['dayofweek'] = data[c].dt.dayofweek
    df['isWeekend'] = (df[c].dt.dayofweek >=5).astype(int)
    df['isWeekday'] = (df[c].dt.dayofweek < 5).astype(int)
    df['dayofyear'] = data[c].dt.dayofyear
    df['quarter'] = data[c].dt.quarter
    df['isMonthStart'] = data[c].dt.is_month_start.astype(int)
    df['isMonthEnd'] = data[c].dt.is_month_end.astype(int)
    df['isQuarterStart'] = data[c].dt.is_quarter_start.astype(int)
    df['isQuarterEnd'] = data[c].dt.is_quarter_end.astype(int)
    df['isYearStart'] = data[c].dt.is_year_start.astype(int)
    df['isYearEnd'] = data[c].dt.is_year_end.astype(int)
    df['isLeapYear'] = data[c].dt.is_leap_year.astype(int)
    df['monthDifferenceFromToday'] = (datetime.datetime.today() - df[c]).dt.days//30
    df['dayDifferenceFromToday'] = (datetime.datetime.today() - df[c]).dt.days
    # Count days until a specific date arrives (value=0 on the day of the event)
    df['blackFriday2017']=(pd.to_datetime('2017-11-24') - df[c]).dt.days.apply(lambda x: x if x >= 0 and x < 100 else -1)

    
def get_session(time):
    if time > 4 and time<12:
        return 0 # Morning
    elif time >= 12 and time < 17:
        return 1 # Afternoon
    elif time >= 17 and time < 21:
        return 2 # Evening
    else:
        return 3 # Night    
```


```python
create_datetime_features(data, 'purchase_date')
```


```python
data.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>card_id</th>
      <th>purchase_date</th>
      <th>city_id</th>
      <th>category_1</th>
      <th>category_2</th>
      <th>category_3</th>
      <th>month_lag</th>
      <th>purchase_amount</th>
      <th>year</th>
      <th>month</th>
      <th>day</th>
      <th>hour</th>
      <th>session</th>
      <th>minute</th>
      <th>second</th>
      <th>week</th>
      <th>dayofweek</th>
      <th>isWeekend</th>
      <th>isWeekday</th>
      <th>dayofyear</th>
      <th>quarter</th>
      <th>isMonthStart</th>
      <th>isMonthEnd</th>
      <th>isQuarterStart</th>
      <th>isQuarterEnd</th>
      <th>isYearStart</th>
      <th>isYearEnd</th>
      <th>isLeapYear</th>
      <th>monthDifferenceFromToday</th>
      <th>dayDifferenceFromToday</th>
      <th>blackFriday2017</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>C_ID_415bb3a509</td>
      <td>2018-03-11 14:57:00</td>
      <td>107</td>
      <td>N</td>
      <td>1.0</td>
      <td>B</td>
      <td>1</td>
      <td>-0.557574</td>
      <td>2018</td>
      <td>3</td>
      <td>11</td>
      <td>14</td>
      <td>1</td>
      <td>57</td>
      <td>0</td>
      <td>10</td>
      <td>6</td>
      <td>1</td>
      <td>0</td>
      <td>70</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>12</td>
      <td>365</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>C_ID_415bb3a509</td>
      <td>2018-03-19 18:53:00</td>
      <td>140</td>
      <td>N</td>
      <td>1.0</td>
      <td>B</td>
      <td>1</td>
      <td>-0.569580</td>
      <td>2018</td>
      <td>3</td>
      <td>19</td>
      <td>18</td>
      <td>2</td>
      <td>53</td>
      <td>0</td>
      <td>12</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>78</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>11</td>
      <td>357</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>C_ID_415bb3a509</td>
      <td>2018-04-26 14:08:00</td>
      <td>330</td>
      <td>N</td>
      <td>1.0</td>
      <td>B</td>
      <td>2</td>
      <td>-0.551037</td>
      <td>2018</td>
      <td>4</td>
      <td>26</td>
      <td>14</td>
      <td>1</td>
      <td>8</td>
      <td>0</td>
      <td>17</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>116</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>10</td>
      <td>319</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>C_ID_415bb3a509</td>
      <td>2018-03-07 09:43:00</td>
      <td>-1</td>
      <td>Y</td>
      <td>NaN</td>
      <td>B</td>
      <td>1</td>
      <td>-0.671926</td>
      <td>2018</td>
      <td>3</td>
      <td>7</td>
      <td>9</td>
      <td>0</td>
      <td>43</td>
      <td>0</td>
      <td>10</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>66</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>12</td>
      <td>369</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>C_ID_ef55cf8d4b</td>
      <td>2018-03-22 21:07:00</td>
      <td>-1</td>
      <td>Y</td>
      <td>NaN</td>
      <td>B</td>
      <td>1</td>
      <td>-0.659904</td>
      <td>2018</td>
      <td>3</td>
      <td>22</td>
      <td>21</td>
      <td>3</td>
      <td>7</td>
      <td>0</td>
      <td>12</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>81</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>11</td>
      <td>354</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>C_ID_ef55cf8d4b</td>
      <td>2018-04-02 12:53:00</td>
      <td>231</td>
      <td>N</td>
      <td>1.0</td>
      <td>B</td>
      <td>2</td>
      <td>-0.633007</td>
      <td>2018</td>
      <td>4</td>
      <td>2</td>
      <td>12</td>
      <td>1</td>
      <td>53</td>
      <td>0</td>
      <td>14</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>92</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>11</td>
      <td>343</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>6</th>
      <td>C_ID_ef55cf8d4b</td>
      <td>2018-03-28 19:50:00</td>
      <td>69</td>
      <td>N</td>
      <td>1.0</td>
      <td>B</td>
      <td>1</td>
      <td>5.263697</td>
      <td>2018</td>
      <td>3</td>
      <td>28</td>
      <td>19</td>
      <td>2</td>
      <td>50</td>
      <td>0</td>
      <td>13</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>87</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>11</td>
      <td>348</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>7</th>
      <td>C_ID_ef55cf8d4b</td>
      <td>2018-04-05 08:06:00</td>
      <td>231</td>
      <td>N</td>
      <td>1.0</td>
      <td>B</td>
      <td>2</td>
      <td>-0.553787</td>
      <td>2018</td>
      <td>4</td>
      <td>5</td>
      <td>8</td>
      <td>0</td>
      <td>6</td>
      <td>0</td>
      <td>14</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>95</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>11</td>
      <td>340</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>8</th>
      <td>C_ID_ef55cf8d4b</td>
      <td>2018-04-07 18:37:00</td>
      <td>69</td>
      <td>N</td>
      <td>1.0</td>
      <td>B</td>
      <td>2</td>
      <td>-0.596643</td>
      <td>2018</td>
      <td>4</td>
      <td>7</td>
      <td>18</td>
      <td>2</td>
      <td>37</td>
      <td>0</td>
      <td>14</td>
      <td>5</td>
      <td>1</td>
      <td>0</td>
      <td>97</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>11</td>
      <td>338</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>9</th>
      <td>C_ID_ef55cf8d4b</td>
      <td>2018-03-17 18:10:00</td>
      <td>69</td>
      <td>N</td>
      <td>1.0</td>
      <td>B</td>
      <td>1</td>
      <td>-0.607191</td>
      <td>2018</td>
      <td>3</td>
      <td>17</td>
      <td>18</td>
      <td>2</td>
      <td>10</td>
      <td>0</td>
      <td>11</td>
      <td>5</td>
      <td>1</td>
      <td>0</td>
      <td>76</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>11</td>
      <td>359</td>
      <td>-1</td>
    </tr>
  </tbody>
</table>
</div>



### Boom!

**23** new features from a single column
