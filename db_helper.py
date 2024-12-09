import mysql.connector
import traceback
from settings import (
    MYSQL_HOST_IP,
    MYSQL_PASSWORD,
    MYSQL_SCHEMA,
    MYSQL_USERNAME,
    TABLE_NAME,
)


def create_db_connection():
    conn = mysql.connector.connect(
        host=MYSQL_HOST_IP,
        user=MYSQL_USERNAME,
        password=MYSQL_PASSWORD,
        db=MYSQL_SCHEMA,
    )
    return conn


def insert_job_in_db(
    job_id,
    doc_id="",
    model_used="",
    blob_path_chart_parser_async="",
    blob_path_chart_parser_v2="",
    blob_path_header_extraction="",
    job_status="",
    chart_parser_v2_status="",
    header_extraction_status="",
    datetime_created=None,
    datetime_chart_finish=None,
    datetime_header_finish=None,
):
    """_summary_

    Args:
        job_id (_type_): job_id of chart job
        doc_id (str, optional): doc_id of chart
        model_used (str, optional): type of job - inhouse,hybrid
        blob_path_chart_parser_async: blob path where chart_parser_async results will be stored
        blob_path_chart_parser_v2 (str, optional): blob path where chart parser v2 results will be stored
        blob_path_header_extraction (str, optional): blob path where header extraction results will be stored
        job_status (str, optional): overall status of chart job.
        job_status_chart_parser (str, optional): status of chart parser v2.
        job_status_header_extraction (str, optional): status of header extraction.
        datetime_created (_type_, optional): timestamp when this job was created
        datetime_chart_finish (_type_, optional): timestamp when chart v2 finished
        datetime_header_finish (_type_, optional): timestamp when header finished

    Returns:
        _type_: _description_
    """
    try:
        conn = create_db_connection()
        ## Cursor runs
        mycursor = conn.cursor()

        ## Execute the query in parametrised fashion
        query = f"INSERT INTO {TABLE_NAME} VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s);"
        values = (
            job_id,
            doc_id,
            model_used,
            blob_path_chart_parser_async,
            blob_path_chart_parser_v2,
            blob_path_header_extraction,
            job_status,
            datetime_created,
            datetime_chart_finish,
            datetime_header_finish,
            chart_parser_v2_status,
            header_extraction_status
        )
        mycursor.execute(query, values)
        ## Commit the conn
        conn.commit()
        mycursor.close()
        conn.close()
        return True, ""
    except Exception as ex:
        print(f"Exception while inserting job {job_id} to mysql")
        traceback_ex = traceback.format_exc()
        return False, traceback_ex


def update_job_from_job_id(job_id, column, status):
    """_summary_
    update a certain column value in a table in mysql
    Args:
        job_id (str): UUID id as str
        column (str): column name to update
        status (str): new status to save
    """
    try:
        conn = create_db_connection()
        ## Cursor runs
        mycursor = conn.cursor()

        ## Execute the query in parametrised fashion
        query = f"UPDATE {TABLE_NAME} SET {column}=(%s) WHERE job_id=(%s);"
        values = (status, job_id)
        mycursor.execute(query, values)
        ## Commit the conn
        conn.commit()
        mycursor.close()
        conn.close()
        return True, ""
    except Exception as ex:
        print(
            f"Exception while updating job {job_id} and column status {column} at mysql"
        )
        traceback_ex = traceback.format_exc()
        return False, traceback_ex


def fetch_job_from_job_id(job_id):
    """_summary_

    Args:
        job_id (str): uuid

    Returns:
        list: all rows from db under certain condition in tuples
    """
    try:
        conn = create_db_connection()
        ## Cursor runs
        mycursor = conn.cursor(dictionary=True)

        ## Execute the query in parametrised fashion
        query = f"Select * from {TABLE_NAME} where job_id=(%s);"
        values = (job_id,)
        mycursor.execute(query, values)
        records = mycursor.fetchall()
        ## Commit the conn
        conn.commit()
        mycursor.close()
        conn.close()
        return True, records
    except Exception as ex:
        print(f"Exception while fetching job {job_id} status row from mysql")
        traceback_ex = traceback.format_exc()
        return False, traceback_ex


def delete_job_from_job_id(job_id):
    try:
        conn = create_db_connection()

        mycursor = conn.cursor()
        query = f"DELETE FROM {TABLE_NAME} WHERE job_id=(%s);"
        values = (job_id,)
        mycursor.execute(query, values)
        ## Commit the conn
        conn.commit()
        mycursor.close()
        conn.close()
        return True, ""
    except Exception as ex:
        print(f"Exception while deleting job {job_id} row from mysql")
        traceback_ex = traceback.format_exc()
        return False, traceback_ex
