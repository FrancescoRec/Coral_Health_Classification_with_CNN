"""
Hello President of IslandCity!
Here you can test the model on the dataset you have.
Just run the script with the name of the dataset as an argument
and you will get the accuracy of the model on the dataset.

Usage:
1. Run the coral_classifier using as a path a folder with inside two folders, one for each class.
2. Name these folder "bleached" and "healthy" and put the images accordingly.
(in this way the id of the image will contain the word "bleached" and "healthy" for the respective folders)
3. Run the test_model_coral.py script with the name of the database as an argument (default is results.db)
4. The script will print the accuracy of the model on the dataset based
    on the matching between the class and the id (that will contain inside the path the value of the class).

es.
ID class match
test_images/bleached/image1.jpg bleached True
test_images/healthy/image2.jpg bleached False


Let me know if you need any help. Be green!
"""

import sqlite3
import pandas as pd
import sys

def test(db):
    conn = sqlite3.connect(db)
    df = pd.read_sql_query("SELECT * FROM sample_results", conn)
    conn.close()

    def match_class(row):
        if row['class'] in row['id']:
            return True
        return False

    df['match'] = df.apply(match_class, axis=1)

    print(f"The model was tested on {df.shape[0]} samples.")
    print(f"The model was accurate for {df['match'].value_counts()[True]} out of {df.shape[0]} samples.")
    print(f"So the accuracy is {df['match'].value_counts()[True] / df.shape[0]}.")

if __name__ == '__main__':
    test(sys.argv[1])


