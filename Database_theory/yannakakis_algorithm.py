import pandas as pd
import glob
import os
import sqlite3
import duckdb

folder="dataset.exc.yan.basic" 

def import_data():
    

    if os.path.isdir(folder):
       
        csv_files = glob.glob(f"{folder}/*.csv", recursive=True)
    
        return csv_files

    else:
        print("Folder dataset.exc.yan.basic with tables is missing!")    
        
        return 0




def yanakakis_algorithm():

    conn = sqlite3.connect(":memory:")  # or "mydb.db" for disk-based DB
    users = pd.read_csv(f"{folder}/users.csv") # users
    posts = pd.read_csv(f"{folder}/posts.csv") # posts
    badges = pd.read_csv(f"{folder}/badges.csv") # badges
    postHistory = pd.read_csv(f"{folder}/postHistory.csv") #postHistory

    posts_filtered = posts[posts['PostTypeId'] == 2]
    ph_filtered = postHistory[(postHistory['CreationDate'] >= '2010-07-27 18:08:19') &(postHistory['CreationDate'] <= '2014-09-10 08:22:43')]

# Reduce users (keep only those that appear in all three tables)
    users_reduced = users[users['Id'].isin(badges['UserId']) &users['Id'].isin(posts_filtered['OwnerUserId']) &users['Id'].isin(ph_filtered['UserId'])]
    badges_filtered = badges[badges['UserId'].isin(users_reduced['Id'])]
    posts_final = posts_filtered[posts_filtered['OwnerUserId'].isin(users_reduced['Id'])]
    ph_final = ph_filtered[ph_filtered['UserId'].isin(users_reduced['Id'])]
    join1 = users_reduced.merge(badges_filtered, left_on='Id', right_on='UserId')
    print(join1)
    join2 = join1.merge(posts_final, left_on='Id_x', right_on='OwnerUserId')
    print(join2)
    final_join = join2.merge(ph_final, left_on='Id_x', right_on='UserId')

    # Count(*) > 1
    more_than_one = len(final_join) > 1
    print("Result:", more_than_one)




def yannakis_algorithm():

  conn = duckdb.connect()

  # Load CSV files into DuckDB tables
  conn.execute("CREATE TABLE users AS SELECT * FROM read_csv_auto('dataset.exc.yan.basic/users.csv')")
  conn.execute("CREATE TABLE posts AS SELECT * FROM read_csv_auto('dataset.exc.yan.basic/posts.csv')")
  conn.execute("CREATE TABLE badges AS SELECT * FROM read_csv_auto('dataset.exc.yan.basic/badges.csv')")
  conn.execute("CREATE TABLE postHistory AS SELECT * FROM read_csv_auto('dataset.exc.yan.basic/postHistory.csv')")

  query = """
SELECT COUNT(*) > 1 AS result
FROM users u
JOIN badges b ON b.UserId = u.Id
JOIN posts p ON p.OwnerUserId = u.Id
JOIN postHistory ph ON ph.UserId = u.Id
WHERE 
    p.PostTypeId = 2
    AND ph.CreationDate BETWEEN '2010-07-27 18:08:19' AND '2014-09-10 08:22:43'
    AND u.Id IN (SELECT UserId FROM badges)
    AND u.Id IN (SELECT OwnerUserId FROM posts WHERE PostTypeId = 2)
    AND u.Id IN (
        SELECT UserId 
        FROM postHistory
        WHERE CreationDate BETWEEN '2010-07-27 18:08:19' AND '2014-09-10 08:22:43'
    );
"""

  result = conn.execute(query).df()
  print(result)
  

  conn.close()



def main():

    csv_files =import_data()

    # Check if csv files exist in folder - unecessary check for this exercise
    # but a good practice in general

    if len(csv_files)>1:
        yannakis_algorithm()




if __name__ == "__main__":
    main()