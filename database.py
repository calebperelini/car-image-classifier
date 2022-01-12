import sqlite3

# connect to db
conn = sqlite3.connect('vehicles.db')

# create cursor
c = conn.cursor()

# create table for plate evaluations.

# c.execute("""CREATE TABLE vehicles (
#         plate_prediction text,
#         score real,
#         processing_time real
#     )"""
#           )

c.execute("INSERT INTO vehicles VALUES ('KNH93', 0.92, 39.23)")

conn.commit()

c.execute("SELECT * FROM vehicles WHERE plate_prediction='KNH93'")

print(c.fetchall())

# commit command, close.
conn.commit()
conn.close()
