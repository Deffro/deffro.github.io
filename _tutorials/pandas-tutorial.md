---
title: "Pandas Tutorial"
tags: [pre-process, pandas]
header:
  image: "/images/tutorials/pandas.jpg"
  teaser: "/images/teasers/pandas.jpg"  
excerpt: "Are you starting with Data Science? Pandas is the first thing you will need. Learn how to create, add, remove, rename, read, select, filter, sort, group, manipulate data."
mathjax: "true"
categories:
  - Data Processing
---

## Pandas Tutorial

Are you starting with Data Science? Pandas is perhaps the first best thing you will need. And it's really easy!

After reading (and practising) this tutorial you will learn how to:

- Create, add, remove and rename columns
- Read, select and filter data
- Retrieve statistics for data
- Sort and group data
- Manipulate data

Happy learning!

### Import Libraries


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)
pd.set_option('max_colwidth', 100)
```

### 1. Rows and Columns (create, add, remove, rename)

#### Create a DataFrame + add rows and columns


```python
data_to_add = {
    'Name':['Greece', 'Paraguay', 'Ireland'],
    'Population':[10.77, 6.62, 4.78],
    'Area':[128.9, 397.3, 68.89]
}
countries = pd.DataFrame(data_to_add, columns=['Name','Population','Area'])
#Note: if you don't specify 'columns', then they will be placed in alphabetical order

countries.head()
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
      <th>Name</th>
      <th>Population</th>
      <th>Area</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Greece</td>
      <td>10.77</td>
      <td>128.90</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Paraguay</td>
      <td>6.62</td>
      <td>397.30</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Ireland</td>
      <td>4.78</td>
      <td>68.89</td>
    </tr>
  </tbody>
</table>
</div>



#### Add data (a row) with *append*


```python
countries = countries.append({'Name':'Japan',
                              'Population':127.25,
                              'Area':364.5}, ignore_index=True)
#Note: If you type the name of a column that does not exist, then it will be created with NaNs for all other rows
countries.head()
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
      <th>Name</th>
      <th>Population</th>
      <th>Area</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Greece</td>
      <td>10.77</td>
      <td>128.90</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Paraguay</td>
      <td>6.62</td>
      <td>397.30</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Ireland</td>
      <td>4.78</td>
      <td>68.89</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Japan</td>
      <td>127.25</td>
      <td>364.50</td>
    </tr>
  </tbody>
</table>
</div>




```python
names = ['Singapore', 'Gabon']
populations = [4.46, 1.64]
areas = [0.687, 257.67]
for i in range(len(names)):
    countries = countries.append({'Name':names[i],
                              'Population':populations[i],
                              'Area':areas[i]}, ignore_index=True)
countries.head(10)
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
      <th>Name</th>
      <th>Population</th>
      <th>Area</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Greece</td>
      <td>10.77</td>
      <td>128.900</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Paraguay</td>
      <td>6.62</td>
      <td>397.300</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Ireland</td>
      <td>4.78</td>
      <td>68.890</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Japan</td>
      <td>127.25</td>
      <td>364.500</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Singapore</td>
      <td>4.46</td>
      <td>0.687</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Gabon</td>
      <td>1.64</td>
      <td>257.670</td>
    </tr>
  </tbody>
</table>
</div>



#### Add feature (a column) with *assign*. Handcrafted data


```python
countries = countries.assign(GDP = [22.08,3.81,45.83,46.72,51.7,11.43])
```

#### Add a feature (a column) from interactions with other features


```python
countries['PopulationPerArea'] = countries['Population']/countries['Area']
countries.head(10)
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
      <th>Name</th>
      <th>Population</th>
      <th>Area</th>
      <th>GDP</th>
      <th>PopulationPerArea</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Greece</td>
      <td>10.77</td>
      <td>128.900</td>
      <td>22.08</td>
      <td>0.083553</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Paraguay</td>
      <td>6.62</td>
      <td>397.300</td>
      <td>3.81</td>
      <td>0.016662</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Ireland</td>
      <td>4.78</td>
      <td>68.890</td>
      <td>45.83</td>
      <td>0.069386</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Japan</td>
      <td>127.25</td>
      <td>364.500</td>
      <td>46.72</td>
      <td>0.349108</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Singapore</td>
      <td>4.46</td>
      <td>0.687</td>
      <td>51.70</td>
      <td>6.491994</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Gabon</td>
      <td>1.64</td>
      <td>257.670</td>
      <td>11.43</td>
      <td>0.006365</td>
    </tr>
  </tbody>
</table>
</div>



#### Remove (drop) a row


```python
countries = countries.drop(3, axis=0) #Drop by index
countries.head(10)
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
      <th>Name</th>
      <th>Population</th>
      <th>Area</th>
      <th>GDP</th>
      <th>PopulationPerArea</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Greece</td>
      <td>10.77</td>
      <td>128.900</td>
      <td>22.08</td>
      <td>0.083553</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Paraguay</td>
      <td>6.62</td>
      <td>397.300</td>
      <td>3.81</td>
      <td>0.016662</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Ireland</td>
      <td>4.78</td>
      <td>68.890</td>
      <td>45.83</td>
      <td>0.069386</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Singapore</td>
      <td>4.46</td>
      <td>0.687</td>
      <td>51.70</td>
      <td>6.491994</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Gabon</td>
      <td>1.64</td>
      <td>257.670</td>
      <td>11.43</td>
      <td>0.006365</td>
    </tr>
  </tbody>
</table>
</div>



#### Remove (drop) a column


```python
countries = countries.drop(['PopulationPerArea'], axis=1)
countries.head(10)
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
      <th>Name</th>
      <th>Population</th>
      <th>Area</th>
      <th>GDP</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Greece</td>
      <td>10.77</td>
      <td>128.900</td>
      <td>22.08</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Paraguay</td>
      <td>6.62</td>
      <td>397.300</td>
      <td>3.81</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Ireland</td>
      <td>4.78</td>
      <td>68.890</td>
      <td>45.83</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Singapore</td>
      <td>4.46</td>
      <td>0.687</td>
      <td>51.70</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Gabon</td>
      <td>1.64</td>
      <td>257.670</td>
      <td>11.43</td>
    </tr>
  </tbody>
</table>
</div>



#### Rename a column


```python
countries = countries.rename(columns={'GDP':'Money'})
countries.head(10)
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
      <th>Name</th>
      <th>Population</th>
      <th>Area</th>
      <th>Money</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Greece</td>
      <td>10.77</td>
      <td>128.900</td>
      <td>22.08</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Paraguay</td>
      <td>6.62</td>
      <td>397.300</td>
      <td>3.81</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Ireland</td>
      <td>4.78</td>
      <td>68.890</td>
      <td>45.83</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Singapore</td>
      <td>4.46</td>
      <td>0.687</td>
      <td>51.70</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Gabon</td>
      <td>1.64</td>
      <td>257.670</td>
      <td>11.43</td>
    </tr>
  </tbody>
</table>
</div>



### 2. Selecting Data 

#### Subsets of columns of DataFrame


```python
print(type(countries['Name'])) # A single column in [] is called Series
countries['Name']
```

    <class 'pandas.core.series.Series'>
    




    0       Greece
    1     Paraguay
    2      Ireland
    4    Singapore
    5        Gabon
    Name: Name, dtype: object




```python
type(countries[['Name']]) # With [[]] it is again a DataFrame
countries[['Name']]
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
      <th>Name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Greece</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Paraguay</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Ireland</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Singapore</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Gabon</td>
    </tr>
  </tbody>
</table>
</div>




```python
countries[['Name','Money']] # More columns
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
      <th>Name</th>
      <th>Money</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Greece</td>
      <td>22.08</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Paraguay</td>
      <td>3.81</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Ireland</td>
      <td>45.83</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Singapore</td>
      <td>51.70</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Gabon</td>
      <td>11.43</td>
    </tr>
  </tbody>
</table>
</div>



#### Subsets of rows of DataFrame with *slice*


```python
countries[0:2] # Return the rows from the 0th to the 1st position
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
      <th>Name</th>
      <th>Population</th>
      <th>Area</th>
      <th>Money</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Greece</td>
      <td>10.77</td>
      <td>128.9</td>
      <td>22.08</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Paraguay</td>
      <td>6.62</td>
      <td>397.3</td>
      <td>3.81</td>
    </tr>
  </tbody>
</table>
</div>



#### Subsets of rows of DataFrame with *iloc*
Same as slice


```python
countries.iloc[0:2]
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
      <th>Name</th>
      <th>Population</th>
      <th>Area</th>
      <th>Money</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Greece</td>
      <td>10.77</td>
      <td>128.9</td>
      <td>22.08</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Paraguay</td>
      <td>6.62</td>
      <td>397.3</td>
      <td>3.81</td>
    </tr>
  </tbody>
</table>
</div>



#### Subsets of rows and columns of DataFrame with iloc


```python
countries.iloc[0:4,0:2]
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
      <th>Name</th>
      <th>Population</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Greece</td>
      <td>10.77</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Paraguay</td>
      <td>6.62</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Ireland</td>
      <td>4.78</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Singapore</td>
      <td>4.46</td>
    </tr>
  </tbody>
</table>
</div>



#### Filter rows and select specific columns


```python
countries[countries['Population']>5][['Name','Population']]
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
      <th>Name</th>
      <th>Population</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Greece</td>
      <td>10.77</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Paraguay</td>
      <td>6.62</td>
    </tr>
  </tbody>
</table>
</div>



#### Select rows of DataFrame with *loc*
loc gets rows (or columns) with particular labels from the index


```python
countries.loc[[1,2]]
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
      <th>Name</th>
      <th>Population</th>
      <th>Area</th>
      <th>Money</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>Paraguay</td>
      <td>6.62</td>
      <td>397.30</td>
      <td>3.81</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Ireland</td>
      <td>4.78</td>
      <td>68.89</td>
      <td>45.83</td>
    </tr>
  </tbody>
</table>
</div>



#### Select rows and columns of DataFrame with loc


```python
countries.loc[[1,2], ['Name','Area']]
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
      <th>Name</th>
      <th>Area</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>Paraguay</td>
      <td>397.30</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Ireland</td>
      <td>68.89</td>
    </tr>
  </tbody>
</table>
</div>



#### Select specific cell (row,column) of a DataFrame with *at*


```python
countries.at[1,'Name'] # index,column_name
```




    'Paraguay'



### 3. Filtering Data

#### Relational Operators


```python
countries[countries['Area']>100]
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
      <th>Name</th>
      <th>Population</th>
      <th>Area</th>
      <th>Money</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Greece</td>
      <td>10.77</td>
      <td>128.90</td>
      <td>22.08</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Paraguay</td>
      <td>6.62</td>
      <td>397.30</td>
      <td>3.81</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Gabon</td>
      <td>1.64</td>
      <td>257.67</td>
      <td>11.43</td>
    </tr>
  </tbody>
</table>
</div>



#### Logical Operators


```python
countries[(countries['Area']>100) & (countries['Population']>5)] # don't forget brackets
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
      <th>Name</th>
      <th>Population</th>
      <th>Area</th>
      <th>Money</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Greece</td>
      <td>10.77</td>
      <td>128.9</td>
      <td>22.08</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Paraguay</td>
      <td>6.62</td>
      <td>397.3</td>
      <td>3.81</td>
    </tr>
  </tbody>
</table>
</div>




```python
countries[countries['Name'].isin(['Greece','Gabon'])]
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
      <th>Name</th>
      <th>Population</th>
      <th>Area</th>
      <th>Money</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Greece</td>
      <td>10.77</td>
      <td>128.90</td>
      <td>22.08</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Gabon</td>
      <td>1.64</td>
      <td>257.67</td>
      <td>11.43</td>
    </tr>
  </tbody>
</table>
</div>



#### If contains string


```python
countries[countries['Name'].str.contains('e')]
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
      <th>Name</th>
      <th>Population</th>
      <th>Area</th>
      <th>Money</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Greece</td>
      <td>10.77</td>
      <td>128.900</td>
      <td>22.08</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Ireland</td>
      <td>4.78</td>
      <td>68.890</td>
      <td>45.83</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Singapore</td>
      <td>4.46</td>
      <td>0.687</td>
      <td>51.70</td>
    </tr>
  </tbody>
</table>
</div>



#### Use query command to find a specific values in a column


```python
countries.query('Area>100 & Population>5')
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
      <th>Name</th>
      <th>Population</th>
      <th>Area</th>
      <th>Money</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Greece</td>
      <td>10.77</td>
      <td>128.9</td>
      <td>22.08</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Paraguay</td>
      <td>6.62</td>
      <td>397.3</td>
      <td>3.81</td>
    </tr>
  </tbody>
</table>
</div>



### 4. Reading Data
House Prices Dataset from https://www.kaggle.com/c/house-prices-advanced-regression-techniques


```python
train = pd.read_csv('train.csv', sep=',')
test = pd.read_csv('test.csv', sep=',')
```

#### Headers of DataFrame


```python
train.columns.values
```




    array(['Id', 'MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street',
           'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig',
           'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',
           'BldgType', 'HouseStyle', 'OverallQual', 'OverallCond',
           'YearBuilt', 'YearRemodAdd', 'RoofStyle', 'RoofMatl',
           'Exterior1st', 'Exterior2nd', 'MasVnrType', 'MasVnrArea',
           'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond',
           'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2',
           'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'Heating', 'HeatingQC',
           'CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF',
           'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',
           'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual',
           'TotRmsAbvGrd', 'Functional', 'Fireplaces', 'FireplaceQu',
           'GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageCars',
           'GarageArea', 'GarageQual', 'GarageCond', 'PavedDrive',
           'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch',
           'ScreenPorch', 'PoolArea', 'PoolQC', 'Fence', 'MiscFeature',
           'MiscVal', 'MoSold', 'YrSold', 'SaleType', 'SaleCondition',
           'SalePrice'], dtype=object)



#### Dimensions of DataFrame


```python
train.shape
```




    (1460, 81)



### 5. Statistics 


```python
train.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1460 entries, 0 to 1459
    Data columns (total 81 columns):
    Id               1460 non-null int64
    MSSubClass       1460 non-null int64
    MSZoning         1460 non-null object
    LotFrontage      1201 non-null float64
    LotArea          1460 non-null int64
    Street           1460 non-null object
    Alley            91 non-null object
    LotShape         1460 non-null object
    LandContour      1460 non-null object
    Utilities        1460 non-null object
    LotConfig        1460 non-null object
    LandSlope        1460 non-null object
    Neighborhood     1460 non-null object
    Condition1       1460 non-null object
    Condition2       1460 non-null object
    BldgType         1460 non-null object
    HouseStyle       1460 non-null object
    OverallQual      1460 non-null int64
    OverallCond      1460 non-null int64
    YearBuilt        1460 non-null int64
    YearRemodAdd     1460 non-null int64
    RoofStyle        1460 non-null object
    RoofMatl         1460 non-null object
    Exterior1st      1460 non-null object
    Exterior2nd      1460 non-null object
    MasVnrType       1452 non-null object
    MasVnrArea       1452 non-null float64
    ExterQual        1460 non-null object
    ExterCond        1460 non-null object
    Foundation       1460 non-null object
    BsmtQual         1423 non-null object
    BsmtCond         1423 non-null object
    BsmtExposure     1422 non-null object
    BsmtFinType1     1423 non-null object
    BsmtFinSF1       1460 non-null int64
    BsmtFinType2     1422 non-null object
    BsmtFinSF2       1460 non-null int64
    BsmtUnfSF        1460 non-null int64
    TotalBsmtSF      1460 non-null int64
    Heating          1460 non-null object
    HeatingQC        1460 non-null object
    CentralAir       1460 non-null object
    Electrical       1459 non-null object
    1stFlrSF         1460 non-null int64
    2ndFlrSF         1460 non-null int64
    LowQualFinSF     1460 non-null int64
    GrLivArea        1460 non-null int64
    BsmtFullBath     1460 non-null int64
    BsmtHalfBath     1460 non-null int64
    FullBath         1460 non-null int64
    HalfBath         1460 non-null int64
    BedroomAbvGr     1460 non-null int64
    KitchenAbvGr     1460 non-null int64
    KitchenQual      1460 non-null object
    TotRmsAbvGrd     1460 non-null int64
    Functional       1460 non-null object
    Fireplaces       1460 non-null int64
    FireplaceQu      770 non-null object
    GarageType       1379 non-null object
    GarageYrBlt      1379 non-null float64
    GarageFinish     1379 non-null object
    GarageCars       1460 non-null int64
    GarageArea       1460 non-null int64
    GarageQual       1379 non-null object
    GarageCond       1379 non-null object
    PavedDrive       1460 non-null object
    WoodDeckSF       1460 non-null int64
    OpenPorchSF      1460 non-null int64
    EnclosedPorch    1460 non-null int64
    3SsnPorch        1460 non-null int64
    ScreenPorch      1460 non-null int64
    PoolArea         1460 non-null int64
    PoolQC           7 non-null object
    Fence            281 non-null object
    MiscFeature      54 non-null object
    MiscVal          1460 non-null int64
    MoSold           1460 non-null int64
    YrSold           1460 non-null int64
    SaleType         1460 non-null object
    SaleCondition    1460 non-null object
    SalePrice        1460 non-null int64
    dtypes: float64(3), int64(35), object(43)
    memory usage: 924.0+ KB
    


```python
train.describe()
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
      <th>Id</th>
      <th>MSSubClass</th>
      <th>LotFrontage</th>
      <th>LotArea</th>
      <th>OverallQual</th>
      <th>OverallCond</th>
      <th>YearBuilt</th>
      <th>YearRemodAdd</th>
      <th>MasVnrArea</th>
      <th>BsmtFinSF1</th>
      <th>BsmtFinSF2</th>
      <th>BsmtUnfSF</th>
      <th>TotalBsmtSF</th>
      <th>1stFlrSF</th>
      <th>2ndFlrSF</th>
      <th>LowQualFinSF</th>
      <th>GrLivArea</th>
      <th>BsmtFullBath</th>
      <th>BsmtHalfBath</th>
      <th>FullBath</th>
      <th>HalfBath</th>
      <th>BedroomAbvGr</th>
      <th>KitchenAbvGr</th>
      <th>TotRmsAbvGrd</th>
      <th>Fireplaces</th>
      <th>GarageYrBlt</th>
      <th>GarageCars</th>
      <th>GarageArea</th>
      <th>WoodDeckSF</th>
      <th>OpenPorchSF</th>
      <th>EnclosedPorch</th>
      <th>3SsnPorch</th>
      <th>ScreenPorch</th>
      <th>PoolArea</th>
      <th>MiscVal</th>
      <th>MoSold</th>
      <th>YrSold</th>
      <th>SalePrice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1201.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1452.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1379.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>730.500000</td>
      <td>56.897260</td>
      <td>70.049958</td>
      <td>10516.828082</td>
      <td>6.099315</td>
      <td>5.575342</td>
      <td>1971.267808</td>
      <td>1984.865753</td>
      <td>103.685262</td>
      <td>443.639726</td>
      <td>46.549315</td>
      <td>567.240411</td>
      <td>1057.429452</td>
      <td>1162.626712</td>
      <td>346.992466</td>
      <td>5.844521</td>
      <td>1515.463699</td>
      <td>0.425342</td>
      <td>0.057534</td>
      <td>1.565068</td>
      <td>0.382877</td>
      <td>2.866438</td>
      <td>1.046575</td>
      <td>6.517808</td>
      <td>0.613014</td>
      <td>1978.506164</td>
      <td>1.767123</td>
      <td>472.980137</td>
      <td>94.244521</td>
      <td>46.660274</td>
      <td>21.954110</td>
      <td>3.409589</td>
      <td>15.060959</td>
      <td>2.758904</td>
      <td>43.489041</td>
      <td>6.321918</td>
      <td>2007.815753</td>
      <td>180921.195890</td>
    </tr>
    <tr>
      <th>std</th>
      <td>421.610009</td>
      <td>42.300571</td>
      <td>24.284752</td>
      <td>9981.264932</td>
      <td>1.382997</td>
      <td>1.112799</td>
      <td>30.202904</td>
      <td>20.645407</td>
      <td>181.066207</td>
      <td>456.098091</td>
      <td>161.319273</td>
      <td>441.866955</td>
      <td>438.705324</td>
      <td>386.587738</td>
      <td>436.528436</td>
      <td>48.623081</td>
      <td>525.480383</td>
      <td>0.518911</td>
      <td>0.238753</td>
      <td>0.550916</td>
      <td>0.502885</td>
      <td>0.815778</td>
      <td>0.220338</td>
      <td>1.625393</td>
      <td>0.644666</td>
      <td>24.689725</td>
      <td>0.747315</td>
      <td>213.804841</td>
      <td>125.338794</td>
      <td>66.256028</td>
      <td>61.119149</td>
      <td>29.317331</td>
      <td>55.757415</td>
      <td>40.177307</td>
      <td>496.123024</td>
      <td>2.703626</td>
      <td>1.328095</td>
      <td>79442.502883</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>20.000000</td>
      <td>21.000000</td>
      <td>1300.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1872.000000</td>
      <td>1950.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>334.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>334.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>1900.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>2006.000000</td>
      <td>34900.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>365.750000</td>
      <td>20.000000</td>
      <td>59.000000</td>
      <td>7553.500000</td>
      <td>5.000000</td>
      <td>5.000000</td>
      <td>1954.000000</td>
      <td>1967.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>223.000000</td>
      <td>795.750000</td>
      <td>882.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1129.500000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>5.000000</td>
      <td>0.000000</td>
      <td>1961.000000</td>
      <td>1.000000</td>
      <td>334.500000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>5.000000</td>
      <td>2007.000000</td>
      <td>129975.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>730.500000</td>
      <td>50.000000</td>
      <td>69.000000</td>
      <td>9478.500000</td>
      <td>6.000000</td>
      <td>5.000000</td>
      <td>1973.000000</td>
      <td>1994.000000</td>
      <td>0.000000</td>
      <td>383.500000</td>
      <td>0.000000</td>
      <td>477.500000</td>
      <td>991.500000</td>
      <td>1087.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1464.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>3.000000</td>
      <td>1.000000</td>
      <td>6.000000</td>
      <td>1.000000</td>
      <td>1980.000000</td>
      <td>2.000000</td>
      <td>480.000000</td>
      <td>0.000000</td>
      <td>25.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>6.000000</td>
      <td>2008.000000</td>
      <td>163000.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1095.250000</td>
      <td>70.000000</td>
      <td>80.000000</td>
      <td>11601.500000</td>
      <td>7.000000</td>
      <td>6.000000</td>
      <td>2000.000000</td>
      <td>2004.000000</td>
      <td>166.000000</td>
      <td>712.250000</td>
      <td>0.000000</td>
      <td>808.000000</td>
      <td>1298.250000</td>
      <td>1391.250000</td>
      <td>728.000000</td>
      <td>0.000000</td>
      <td>1776.750000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>1.000000</td>
      <td>7.000000</td>
      <td>1.000000</td>
      <td>2002.000000</td>
      <td>2.000000</td>
      <td>576.000000</td>
      <td>168.000000</td>
      <td>68.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>8.000000</td>
      <td>2009.000000</td>
      <td>214000.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1460.000000</td>
      <td>190.000000</td>
      <td>313.000000</td>
      <td>215245.000000</td>
      <td>10.000000</td>
      <td>9.000000</td>
      <td>2010.000000</td>
      <td>2010.000000</td>
      <td>1600.000000</td>
      <td>5644.000000</td>
      <td>1474.000000</td>
      <td>2336.000000</td>
      <td>6110.000000</td>
      <td>4692.000000</td>
      <td>2065.000000</td>
      <td>572.000000</td>
      <td>5642.000000</td>
      <td>3.000000</td>
      <td>2.000000</td>
      <td>3.000000</td>
      <td>2.000000</td>
      <td>8.000000</td>
      <td>3.000000</td>
      <td>14.000000</td>
      <td>3.000000</td>
      <td>2010.000000</td>
      <td>4.000000</td>
      <td>1418.000000</td>
      <td>857.000000</td>
      <td>547.000000</td>
      <td>552.000000</td>
      <td>508.000000</td>
      <td>480.000000</td>
      <td>738.000000</td>
      <td>15500.000000</td>
      <td>12.000000</td>
      <td>2010.000000</td>
      <td>755000.000000</td>
    </tr>
  </tbody>
</table>
</div>



#### mean, median, quantile, min, max


```python
print('Year Build mean: {:.2f}'.format(train['YearBuilt'].mean()))
print('Year Build median: {:.2f}'.format(train['YearBuilt'].median()))
print('Year Build quantile 25%: {:.2f}'.format(train['YearBuilt'].quantile(0.25)))
print('Year Build quantile 70%: {:.2f}'.format(train['YearBuilt'].quantile(0.7)))
print('Year Build min: {:.2f}'.format(train['YearBuilt'].min()))
print('Year Build max: {:.2f}'.format(train['YearBuilt'].max()))
```

    Year Build mean: 1971.27
    Year Build median: 1973.00
    Year Build quantile 25%: 1954.00
    Year Build quantile 70%: 1997.30
    Year Build min: 1872.00
    Year Build max: 2010.00
    

#### Find the value of the index that has max


```python
train['YearBuilt'].idxmax()
```




    378



#### Count the different occurencies of a column


```python
train['Fireplaces'].value_counts()
```




    0    690
    1    650
    2    115
    3      5
    Name: Fireplaces, dtype: int64



#### Find unique instances of a column


```python
print(train['Fireplaces'].nunique()) #how many unique
train['Fireplaces'].unique() #displays the unique
```

    4
    




    array([0, 1, 2, 3], dtype=int64)



#### Count


```python
train['YearBuilt'].count()
```




    1460



#### Sum


```python
train['SalePrice'].sum()
```




    264144946



#### Mode (Most common value)


```python
train['YearBuilt'].mode()
```




    0    2006
    dtype: int64



### 6. Sorting and Grouping Data 

#### Sort


```python
train.sort_values(by='SalePrice', ascending=False).head()
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
      <th>Id</th>
      <th>MSSubClass</th>
      <th>MSZoning</th>
      <th>LotFrontage</th>
      <th>LotArea</th>
      <th>Street</th>
      <th>Alley</th>
      <th>LotShape</th>
      <th>LandContour</th>
      <th>Utilities</th>
      <th>LotConfig</th>
      <th>LandSlope</th>
      <th>Neighborhood</th>
      <th>Condition1</th>
      <th>Condition2</th>
      <th>BldgType</th>
      <th>HouseStyle</th>
      <th>OverallQual</th>
      <th>OverallCond</th>
      <th>YearBuilt</th>
      <th>YearRemodAdd</th>
      <th>RoofStyle</th>
      <th>RoofMatl</th>
      <th>Exterior1st</th>
      <th>Exterior2nd</th>
      <th>MasVnrType</th>
      <th>MasVnrArea</th>
      <th>ExterQual</th>
      <th>ExterCond</th>
      <th>Foundation</th>
      <th>BsmtQual</th>
      <th>BsmtCond</th>
      <th>BsmtExposure</th>
      <th>BsmtFinType1</th>
      <th>BsmtFinSF1</th>
      <th>BsmtFinType2</th>
      <th>BsmtFinSF2</th>
      <th>BsmtUnfSF</th>
      <th>TotalBsmtSF</th>
      <th>Heating</th>
      <th>HeatingQC</th>
      <th>CentralAir</th>
      <th>Electrical</th>
      <th>1stFlrSF</th>
      <th>2ndFlrSF</th>
      <th>LowQualFinSF</th>
      <th>GrLivArea</th>
      <th>BsmtFullBath</th>
      <th>BsmtHalfBath</th>
      <th>FullBath</th>
      <th>HalfBath</th>
      <th>BedroomAbvGr</th>
      <th>KitchenAbvGr</th>
      <th>KitchenQual</th>
      <th>TotRmsAbvGrd</th>
      <th>Functional</th>
      <th>Fireplaces</th>
      <th>FireplaceQu</th>
      <th>GarageType</th>
      <th>GarageYrBlt</th>
      <th>GarageFinish</th>
      <th>GarageCars</th>
      <th>GarageArea</th>
      <th>GarageQual</th>
      <th>GarageCond</th>
      <th>PavedDrive</th>
      <th>WoodDeckSF</th>
      <th>OpenPorchSF</th>
      <th>EnclosedPorch</th>
      <th>3SsnPorch</th>
      <th>ScreenPorch</th>
      <th>PoolArea</th>
      <th>PoolQC</th>
      <th>Fence</th>
      <th>MiscFeature</th>
      <th>MiscVal</th>
      <th>MoSold</th>
      <th>YrSold</th>
      <th>SaleType</th>
      <th>SaleCondition</th>
      <th>SalePrice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>691</th>
      <td>692</td>
      <td>60</td>
      <td>RL</td>
      <td>104.0</td>
      <td>21535</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Corner</td>
      <td>Gtl</td>
      <td>NoRidge</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>2Story</td>
      <td>10</td>
      <td>6</td>
      <td>1994</td>
      <td>1995</td>
      <td>Gable</td>
      <td>WdShngl</td>
      <td>HdBoard</td>
      <td>HdBoard</td>
      <td>BrkFace</td>
      <td>1170.0</td>
      <td>Ex</td>
      <td>TA</td>
      <td>PConc</td>
      <td>Ex</td>
      <td>TA</td>
      <td>Gd</td>
      <td>GLQ</td>
      <td>1455</td>
      <td>Unf</td>
      <td>0</td>
      <td>989</td>
      <td>2444</td>
      <td>GasA</td>
      <td>Ex</td>
      <td>Y</td>
      <td>SBrkr</td>
      <td>2444</td>
      <td>1872</td>
      <td>0</td>
      <td>4316</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>Ex</td>
      <td>10</td>
      <td>Typ</td>
      <td>2</td>
      <td>Ex</td>
      <td>Attchd</td>
      <td>1994.0</td>
      <td>Fin</td>
      <td>3</td>
      <td>832</td>
      <td>TA</td>
      <td>TA</td>
      <td>Y</td>
      <td>382</td>
      <td>50</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>1</td>
      <td>2007</td>
      <td>WD</td>
      <td>Normal</td>
      <td>755000</td>
    </tr>
    <tr>
      <th>1182</th>
      <td>1183</td>
      <td>60</td>
      <td>RL</td>
      <td>160.0</td>
      <td>15623</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Corner</td>
      <td>Gtl</td>
      <td>NoRidge</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>2Story</td>
      <td>10</td>
      <td>5</td>
      <td>1996</td>
      <td>1996</td>
      <td>Hip</td>
      <td>CompShg</td>
      <td>Wd Sdng</td>
      <td>ImStucc</td>
      <td>None</td>
      <td>0.0</td>
      <td>Gd</td>
      <td>TA</td>
      <td>PConc</td>
      <td>Ex</td>
      <td>TA</td>
      <td>Av</td>
      <td>GLQ</td>
      <td>2096</td>
      <td>Unf</td>
      <td>0</td>
      <td>300</td>
      <td>2396</td>
      <td>GasA</td>
      <td>Ex</td>
      <td>Y</td>
      <td>SBrkr</td>
      <td>2411</td>
      <td>2065</td>
      <td>0</td>
      <td>4476</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>Ex</td>
      <td>10</td>
      <td>Typ</td>
      <td>2</td>
      <td>TA</td>
      <td>Attchd</td>
      <td>1996.0</td>
      <td>Fin</td>
      <td>3</td>
      <td>813</td>
      <td>TA</td>
      <td>TA</td>
      <td>Y</td>
      <td>171</td>
      <td>78</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>555</td>
      <td>Ex</td>
      <td>MnPrv</td>
      <td>NaN</td>
      <td>0</td>
      <td>7</td>
      <td>2007</td>
      <td>WD</td>
      <td>Abnorml</td>
      <td>745000</td>
    </tr>
    <tr>
      <th>1169</th>
      <td>1170</td>
      <td>60</td>
      <td>RL</td>
      <td>118.0</td>
      <td>35760</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>CulDSac</td>
      <td>Gtl</td>
      <td>NoRidge</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>2Story</td>
      <td>10</td>
      <td>5</td>
      <td>1995</td>
      <td>1996</td>
      <td>Hip</td>
      <td>CompShg</td>
      <td>HdBoard</td>
      <td>HdBoard</td>
      <td>BrkFace</td>
      <td>1378.0</td>
      <td>Gd</td>
      <td>Gd</td>
      <td>PConc</td>
      <td>Ex</td>
      <td>TA</td>
      <td>Gd</td>
      <td>GLQ</td>
      <td>1387</td>
      <td>Unf</td>
      <td>0</td>
      <td>543</td>
      <td>1930</td>
      <td>GasA</td>
      <td>Ex</td>
      <td>Y</td>
      <td>SBrkr</td>
      <td>1831</td>
      <td>1796</td>
      <td>0</td>
      <td>3627</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>Gd</td>
      <td>10</td>
      <td>Typ</td>
      <td>1</td>
      <td>TA</td>
      <td>Attchd</td>
      <td>1995.0</td>
      <td>Fin</td>
      <td>3</td>
      <td>807</td>
      <td>TA</td>
      <td>TA</td>
      <td>Y</td>
      <td>361</td>
      <td>76</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>7</td>
      <td>2006</td>
      <td>WD</td>
      <td>Normal</td>
      <td>625000</td>
    </tr>
    <tr>
      <th>898</th>
      <td>899</td>
      <td>20</td>
      <td>RL</td>
      <td>100.0</td>
      <td>12919</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>Gtl</td>
      <td>NridgHt</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>1Story</td>
      <td>9</td>
      <td>5</td>
      <td>2009</td>
      <td>2010</td>
      <td>Hip</td>
      <td>CompShg</td>
      <td>VinylSd</td>
      <td>VinylSd</td>
      <td>Stone</td>
      <td>760.0</td>
      <td>Ex</td>
      <td>TA</td>
      <td>PConc</td>
      <td>Ex</td>
      <td>TA</td>
      <td>Gd</td>
      <td>GLQ</td>
      <td>2188</td>
      <td>Unf</td>
      <td>0</td>
      <td>142</td>
      <td>2330</td>
      <td>GasA</td>
      <td>Ex</td>
      <td>Y</td>
      <td>SBrkr</td>
      <td>2364</td>
      <td>0</td>
      <td>0</td>
      <td>2364</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>Ex</td>
      <td>11</td>
      <td>Typ</td>
      <td>2</td>
      <td>Gd</td>
      <td>Attchd</td>
      <td>2009.0</td>
      <td>Fin</td>
      <td>3</td>
      <td>820</td>
      <td>TA</td>
      <td>TA</td>
      <td>Y</td>
      <td>0</td>
      <td>67</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>3</td>
      <td>2010</td>
      <td>New</td>
      <td>Partial</td>
      <td>611657</td>
    </tr>
    <tr>
      <th>803</th>
      <td>804</td>
      <td>60</td>
      <td>RL</td>
      <td>107.0</td>
      <td>13891</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>Gtl</td>
      <td>NridgHt</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>2Story</td>
      <td>9</td>
      <td>5</td>
      <td>2008</td>
      <td>2009</td>
      <td>Hip</td>
      <td>CompShg</td>
      <td>VinylSd</td>
      <td>VinylSd</td>
      <td>Stone</td>
      <td>424.0</td>
      <td>Ex</td>
      <td>TA</td>
      <td>PConc</td>
      <td>Ex</td>
      <td>TA</td>
      <td>Gd</td>
      <td>Unf</td>
      <td>0</td>
      <td>Unf</td>
      <td>0</td>
      <td>1734</td>
      <td>1734</td>
      <td>GasA</td>
      <td>Ex</td>
      <td>Y</td>
      <td>SBrkr</td>
      <td>1734</td>
      <td>1088</td>
      <td>0</td>
      <td>2822</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>Ex</td>
      <td>12</td>
      <td>Typ</td>
      <td>1</td>
      <td>Gd</td>
      <td>BuiltIn</td>
      <td>2009.0</td>
      <td>RFn</td>
      <td>3</td>
      <td>1020</td>
      <td>TA</td>
      <td>TA</td>
      <td>Y</td>
      <td>52</td>
      <td>170</td>
      <td>0</td>
      <td>0</td>
      <td>192</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>1</td>
      <td>2009</td>
      <td>New</td>
      <td>Partial</td>
      <td>582933</td>
    </tr>
  </tbody>
</table>
</div>




```python
train.sort_values(['Fireplaces', 'SalePrice'], ascending=[0,1])[['Fireplaces', 'SalePrice']].head(8) # 0 means ascending=False
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
      <th>Fireplaces</th>
      <th>SalePrice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1298</th>
      <td>3</td>
      <td>160000</td>
    </tr>
    <tr>
      <th>166</th>
      <td>3</td>
      <td>190000</td>
    </tr>
    <tr>
      <th>605</th>
      <td>3</td>
      <td>205000</td>
    </tr>
    <tr>
      <th>642</th>
      <td>3</td>
      <td>345000</td>
    </tr>
    <tr>
      <th>309</th>
      <td>3</td>
      <td>360000</td>
    </tr>
    <tr>
      <th>393</th>
      <td>2</td>
      <td>100000</td>
    </tr>
    <tr>
      <th>662</th>
      <td>2</td>
      <td>110000</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2</td>
      <td>118000</td>
    </tr>
  </tbody>
</table>
</div>



#### Group


```python
train.groupby('Fireplaces')['SalePrice'].mean()
```




    Fireplaces
    0    141331.482609
    1    211843.909231
    2    240588.539130
    3    252000.000000
    Name: SalePrice, dtype: float64



#### Group continious variables to distinct classes with *cut*


```python
train['YearBuilt_Distinct'] = pd.cut(train['YearBuilt'], [-1, 1950, 1980, 2000, 2010], labels=['Before 1951','1951-1980', '1981-1999', '2001-2010'])
train['YearBuilt_Distinct'].head()
```




    0      2001-2010
    1      1951-1980
    2      2001-2010
    3    Before 1951
    4      1981-1999
    Name: YearBuilt_Distinct, dtype: category
    Categories (4, object): [Before 1951 < 1951-1980 < 1981-1999 < 2001-2010]



### 7. Manipulating Data

#### Transform
Applies to groupby. The output is the same shape as the input dataframe. The grouped values are placed in each corresponding row.


```python
train['meanLivAreaPerYearBuilt'] = train.groupby('YearBuilt')['GrLivArea'].transform('mean')
train[['YearBuilt','meanLivAreaPerYearBuilt']][25:30] # See that same years have the same value
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
      <th>YearBuilt</th>
      <th>meanLivAreaPerYearBuilt</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>25</th>
      <td>2007</td>
      <td>1721.448980</td>
    </tr>
    <tr>
      <th>26</th>
      <td>1951</td>
      <td>1239.166667</td>
    </tr>
    <tr>
      <th>27</th>
      <td>2007</td>
      <td>1721.448980</td>
    </tr>
    <tr>
      <th>28</th>
      <td>1957</td>
      <td>1379.850000</td>
    </tr>
    <tr>
      <th>29</th>
      <td>1927</td>
      <td>838.666667</td>
    </tr>
  </tbody>
</table>
</div>



#### Map
Works with Series (one column). Use it with a *lambda* function and think of it like a for loop. For each row in column 'GrLivArea', run a function


```python
train['GrLivArea'].map(lambda x: x/train['GrLivArea'].max()).head()
```




    0    0.303084
    1    0.223680
    2    0.316554
    3    0.304325
    4    0.389578
    Name: GrLivArea, dtype: float64



#### Apply
Works with DataFrame (one or more columns). Use it with *lambda* like map


```python
train[['YearBuilt','GrLivArea']].apply(lambda x: x - x.min()).head()
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
      <th>YearBuilt</th>
      <th>GrLivArea</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>131</td>
      <td>1376</td>
    </tr>
    <tr>
      <th>1</th>
      <td>104</td>
      <td>928</td>
    </tr>
    <tr>
      <th>2</th>
      <td>129</td>
      <td>1452</td>
    </tr>
    <tr>
      <th>3</th>
      <td>43</td>
      <td>1383</td>
    </tr>
    <tr>
      <th>4</th>
      <td>128</td>
      <td>1864</td>
    </tr>
  </tbody>
</table>
</div>



Or use it to get single values as answer to a function


```python
train[['YearBuilt','GrLivArea']].apply(lambda x: x.max() - x.min())
```




    YearBuilt     138
    GrLivArea    5308
    dtype: int64



#### List Comprehension


```python
train['GrLivArea_binary'] = ['big' if i>1500 else 'normal' for i in train['GrLivArea']]
train[['GrLivArea','GrLivArea_binary']].head()
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
      <th>GrLivArea</th>
      <th>GrLivArea_binary</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1710</td>
      <td>big</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1262</td>
      <td>normal</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1786</td>
      <td>big</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1717</td>
      <td>big</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2198</td>
      <td>big</td>
    </tr>
  </tbody>
</table>
</div>



**Change/Replace** values based on condition


```python
train.loc[train['GrLivArea'] > 2000,  'GrLivArea'] = 2000
```
