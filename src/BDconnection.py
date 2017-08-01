
import pymysql.cursors
import pymysql
from pandas.io import sql
import pandas as pd
from numba import jit


# Connect to the database
connection = pymysql.connect(host='localhost',
                             user='root',
                             password='vader',
                             db='am',
                             charset='utf8mb4',
                             cursorclass=pymysql.cursors.DictCursor)

@jit
def saveData(data,table):
    #(frame, name, con, flavor, schema, if_exists, index, index_label, chunksize, dtype)
    sql.to_sql(data, name=table, con=connection, 
               flavor='mysql', if_exists='append',index=False)
    return 'salvo'

@jit
def getData(table):
    sql = "SELECT * FROM %s" % table
    return pd.read_sql_query(sql, connection)
    
