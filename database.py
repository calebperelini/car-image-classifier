"""
For SQLite DB handling.
"""
import sqlite3

# initialise db, create data table.


def init_db():
    con = sqlite3.connect('vehicles.db')
    c = con.cursor()
    c.execute("""CREATE TABLE vehicles (
        plate_prediction text,
        score real,
        processing_time real
    )"""
              )
    print("Database initialised successfully.")
    con.commit()
    con.close()


def add_one(el):
    con = sqlite3.connect('vehicles.db')
    c = con.cursor()
    c.execute("INSERT INTO vehicles VALUES (?, ?, ?)",
              (el['plate'], el['confidence'], el['process_time']))
    con.commit()
    con.close()


def del_by_key(id):
    con = sqlite3.connect('vehicles.db')
    c = con.cursor()
    c.execute("DELETE from customers WHERE rowid = (?)", id)
    con.commit()
    con.close()


def add_many(elements):
    con = sqlite3.connect('vehicles.db')
    c = con.cursor()
    c.executemany("INSERT INTO vehicles VALUES (?, ?, ?)",
                  (elements))
    con.commit()
    con.close()


# Query db, return records.
def show_all():
    con = sqlite3.connect('vehicles.db')
    c = con.cursor()
    c.execute("SELECT rowid, * FROM vehicles")
    items = c.fetchall()
    for i in items:
        print(i)

    con.commit
    con.close
