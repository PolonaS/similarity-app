import sqlite3

conn = None


def create_connection():
    return sqlite3.connect('similarity.sqlite')

def create_table():
    global conn

    sql = """
        CREATE TABLE IF NOT EXISTS 
        acronyms(
            id integer PRIMARY KEY AUTOINCREMENT, 
            acronym_document_id integer,
            acronym text,
            acronym_context text, 
            full_form_document_id integer,
            full_form text,
            full_form_context text,
            own_cosine_similarity decimal(16,12),
            sklearn_cosine_similarity decimal(16,12),
            own_jaccard_similarity decimal(16,12),
            sklearn_jaccard_similarity decimal(16,12)
        )
    """

    try:
        if conn is None:
            conn = create_connection()
        c = conn.cursor()
        c.execute(sql)
        c.execute("DELETE FROM acronyms")
        conn.commit()
        c.close()
    except sqlite3.Error as e:
        print(e)
    except Exception as e:
        print(e)


def insert(acronym_document_id,
           acronym,
           acronym_context,
           full_form_document_id,
           full_form,
           full_form_context,
           own_cosine_similarity,
           sklearn_cosine_similarity,
           own_jaccard_similarity,
           sklearn_jaccard_similarity):
    global conn

    sql = """
        INSERT INTO 
        acronyms(
            acronym_document_id, 
            acronym,
            acronym_context, 
            full_form_document_id,
            full_form,
            full_form_context,
            own_cosine_similarity,
            sklearn_cosine_similarity,
            own_jaccard_similarity,
            sklearn_jaccard_similarity
        )
        VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """

    try:
        if conn is None:
            conn = create_connection()
        cur = conn.cursor()
        cur.execute(sql, [
            acronym_document_id,
            acronym,
            acronym_context,
            full_form_document_id,
            full_form,
            full_form_context,
            own_cosine_similarity,
            sklearn_cosine_similarity,
            own_jaccard_similarity,
            sklearn_jaccard_similarity
        ])
        conn.commit()
        cur.close()
    except sqlite3.Error as e:
        print(e)
    except Exception as e:
        print(e)
